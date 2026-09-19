from typing import Optional, Callable

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from config import RunConfig
from constants import *
from models.stable_diffusion import CrossImageAttentionStableDiffusionPipeline
from utils import attention_utils
from utils.adain import adain, custom_adain_pixel
from utils.model_utils import get_stable_diffusion_model


class CACTIFModel:
    """
    CACTIFModel handles class-wise AdaIN during generation and a
    custom Stable Diffusion pipeline with Selective Attention Filtering.
    """

    def __init__(self, config: RunConfig, pipe: Optional[CrossImageAttentionStableDiffusionPipeline] = None):
        self.config = config
        self.pipe = get_stable_diffusion_model() if pipe is None else pipe
        self.register_attention_control()

        self.latents_style, self.latents_content = None, None
        self.zs_style, self.zs_content = None, None

        self.label_content = None
        self.label_style = []
        self.feat_style = []

        self.label_style_adain, self.label_content_adain = None, None
        self.style_mu, self.style_sigma = None, None
        self.onehot_mask_content, self.onehot_mask_style = None, None

        self.enable_edit = False
        self.step = 0

    def set_onehot_masks(self):
        """
        Generates class-based one-hot segmentation masks for content and style labels.
        """
        n_classes = 20
        onehot_mask_content = np.zeros((n_classes, *self.label_content.shape), dtype=np.uint8)
        onehot_mask_style = np.zeros((n_classes, *self.label_style[0].shape), dtype=np.uint8)

        for i in range(n_classes):
            onehot_mask_content[i] = (self.label_content == i).astype(np.uint8)
            onehot_mask_style[i] = (self.label_style[0] == i).astype(np.uint8)

        # Handle undefined class (label 255)
        onehot_mask_content[-1] = (self.label_content == 255).astype(np.uint8)
        onehot_mask_style[-1] = (self.label_style[0] == 255).astype(np.uint8)

        self.onehot_mask_content = torch.from_numpy(onehot_mask_content).view(1, 1, n_classes, 512, 1024)
        self.onehot_mask_style = torch.from_numpy(onehot_mask_style).view(1, 1, n_classes, 512, 1024)

    # Latents setter
    def set_latents(self, latents_style: torch.Tensor, latents_content: torch.Tensor):
        self.latents_style = latents_style
        self.latents_content = latents_content

    # Noise setter
    def set_noise(self, zs_style: torch.Tensor, zs_content: torch.Tensor):
        self.zs_style = zs_style
        self.zs_content = zs_content

    def get_adain_callback(self) -> Callable:
        """
        Returns a callback function for AdaIN or class-AdaIN based on the current step and config.
        """
        def callback(st: int, t: int, latents: torch.FloatTensor) -> None:
            self.step = st

            if self.config.class_adain_range.start <= self.step < self.config.class_adain_range.end and self.config.adain_class:
                # Apply class-wise AdaIN
                latents[0] = custom_adain_pixel(latents[0], latents[1], self.label_content_adain, self.label_style_adain)
            else:
                # Apply standard AdaIN
                latents[0] = adain(latents[0], latents[1])
        
        return callback

    def register_attention_control(self):
        """
        Registers a custom attention control mechanism that modifies cross-attention maps
        by selectively applying cross-attention based on feature similarity.
        """
        model_self = self

        class AttentionProcessor:
            def __init__(self, place_in_unet: str):
                self.place_in_unet = place_in_unet

                if not hasattr(F, "scaled_dot_product_attention"):
                    raise ImportError(
                        "AttentionProcessor requires torch 2.0+. Please upgrade your torch installation."
                    )

            def attention_filtering(self, model_self, a_out, a_content, V):
                prc = model_self.config.filter_perc

                max_idx = a_out.sum(dim=0).argmax(dim=-1)                      # [N]; abs() was a no-op (weights >= 0)
                v_content = V[CONTENT_INDEX]                                    # [H,N,D]
                v_style_at_max = V[STYLE_INDEX][:, max_idx]                     # [H,N,D]

                cos = F.cosine_similarity(v_content.float(), v_style_at_max.float(), dim=-1, eps=1e-6)
                score = cos.abs().sum(dim=0)                                    # [N]
                weak = score < torch.quantile(score, prc)                       # [N] bool, stays on GPU

                a_out = torch.where(weak, a_content, a_out)                     # mask over key dim
                v_out = torch.where(weak[:, None], v_content, V[OUT_INDEX])
                return a_out, v_out

            def __call__(self,
                         attn,
                         hidden_states: torch.Tensor,
                         encoder_hidden_states: Optional[torch.Tensor] = None,
                         attention_mask=None,
                         temb=None,
                         perform_swap: bool = False):
                
                residual = hidden_states

                if attn.spatial_norm is not None:
                    hidden_states = attn.spatial_norm(hidden_states, temb)

                input_ndim = hidden_states.ndim
                
                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

                batch_size, sequence_length, _ = (
                    hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                )

                if attention_mask is not None:
                    attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                    attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

                if attn.group_norm is not None:
                    hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                    
                is_cross = encoder_hidden_states is not None
                
                query = attn.to_q(hidden_states)
                
                if not is_cross:
                    encoder_hidden_states = hidden_states
                elif attn.norm_cross:
                    encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

                key = attn.to_k(encoder_hidden_states)
                value = attn.to_v(encoder_hidden_states)
                
                inner_dim = key.shape[-1]
                head_dim = inner_dim // attn.heads
                should_mix = False
                
                # Potentially apply cross image attention operation
                # To do so, we need to be in a self-attention layer in the decoder part of the denoising network
                if model_self.config.cross_attention:
                    if perform_swap and not is_cross and "up" in self.place_in_unet and model_self.enable_edit:
                        if attention_utils.should_mix_keys_and_values(model_self, hidden_states):
                            should_mix = True
                            # Inject the appearance's keys and values
                            key[OUT_INDEX] = key[STYLE_INDEX]
                            value[OUT_INDEX] = value[STYLE_INDEX]

                query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                
                do_edit = perform_swap and model_self.enable_edit and should_mix and not is_cross
                use_filter = do_edit and model_self.config.filtering

                hidden_states, maps = attention_utils.compute_scaled_dot_product_attention(
                    query, key, value,
                    edit_map=do_edit,
                    is_cross=is_cross,
                    contrast_strength=model_self.config.contrast_strength,
                    return_maps=use_filter,
                )

                if use_filter:
                    a_out, v_out = self.attention_filtering(model_self, *maps, value)
                    hidden_states[OUT_INDEX] = a_out @ v_out      # only the OUT element changes
                
                      
                hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
                hidden_states = hidden_states.to(query[OUT_INDEX].dtype)

                # linear proj
                hidden_states = attn.to_out[0](hidden_states)
                # dropout
                hidden_states = attn.to_out[1](hidden_states)

                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

                if attn.residual_connection:
                    hidden_states = hidden_states + residual

                hidden_states = hidden_states / attn.rescale_output_factor
                
                return hidden_states

        def register_recr(net_, count, place_in_unet):
            if net_.__class__.__name__ == 'ResnetBlock2D':
                pass
            if net_.__class__.__name__ == 'Attention':
                net_.set_processor(AttentionProcessor(place_in_unet + f"_{count + 1}"))
                return count + 1
            elif hasattr(net_, 'children'):
                for net__ in net_.children():
                    count = register_recr(net__, count, place_in_unet)
            return count

        cross_att_count = 0
        sub_nets = self.pipe.unet.named_children()
        for net in sub_nets:
            if "down" in net[0]:
                cross_att_count += register_recr(net[1], 0, "down")
            elif "up" in net[0]:
                cross_att_count += register_recr(net[1], 0, "up")
            elif "mid" in net[0]:
                cross_att_count += register_recr(net[1], 0, "mid")
