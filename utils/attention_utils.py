import math
import torch

from constants import *


def should_mix_keys_and_values(model, hidden_states: torch.Tensor) -> bool:
    """ Verify whether we should perform the mixing in the current timestep. """
    is_in_32_timestep_range = (
            model.config.cross_attn_32_range.start <= model.step < model.config.cross_attn_32_range.end
    )
    is_in_64_timestep_range = (
            model.config.cross_attn_64_range.start <= model.step < model.config.cross_attn_64_range.end
    )

    is_hidden_states_32 = (hidden_states.shape[1] == 32*64) # (hidden_states.shape[1] == 32 ** 2) if square input image
    is_hidden_states_64 = (hidden_states.shape[1] == 64*128) # (hidden_states.shape[1] == 64 ** 2) if square input image
    should_mix = (is_in_64_timestep_range and is_hidden_states_64) or \
                (is_in_32_timestep_range and is_hidden_states_32)

    return should_mix

import torch.nn.functional as F

def compute_scaled_dot_product_attention(Q, K, V, edit_map=False, is_cross=False,
                                         contrast_strength=1.0, return_maps=False):
    """
    Q, K, V: [B, H, N, D].
    Returns (hidden_states, maps). `maps` is None unless return_maps=True in an edit layer,
    in which case it is (attn_out, attn_content), each [H, N, N].
    """
    # Fused flash / mem-efficient kernel for all batch elements: never builds the N x N matrix.
    hidden = F.scaled_dot_product_attention(Q, K, V)

    if not (edit_map and not is_cross):
        return hidden, None

    # Explicit maps only for the elements that need them.
    scale = 1.0 / math.sqrt(Q.size(-1))
    a_out = torch.softmax((Q[OUT_INDEX] @ K[OUT_INDEX].transpose(-2, -1)) * scale, dim=-1)  # [H,N,N]

    # Contrast, vectorised over heads (same broadcasting as your enhance_tensor)
    mu = a_out.mean(dim=-1).unsqueeze(-2)                                                    # [H,1,N]
    a_out = a_out.sub_(mu).mul_(contrast_strength).add_(mu).clamp_(0.0, 1.0)

    if return_maps:
        #a_content = torch.softmax(
            #(Q[CONTENT_INDEX] @ K[CONTENT_INDEX].transpose(-2, -1)) * scale, dim=-1
        #)
        return hidden, a_out        # a_out: [H,N,N], contrast-enhanced cross-image attention

    hidden[OUT_INDEX] = a_out @ V[OUT_INDEX]
    return hidden, None


def enhance_tensor(tensor: torch.Tensor, contrast_factor: float = 1.67) -> torch.Tensor:
    """ Compute the attention map contrasting. """
    adjusted_tensor = (tensor - tensor.mean(dim=-1)) * contrast_factor + tensor.mean(dim=-1)
    return adjusted_tensor
