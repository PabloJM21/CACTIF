import torch
from diffusers import DDIMScheduler

from models.stable_diffusion import CrossImageAttentionStableDiffusionPipeline
from models.unet_2d_condition import FreeUUNet2DConditionModel


def get_stable_diffusion_model():
    print("Loading Stable Diffusion model...")
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    dtype = torch.float16 if cuda else torch.float32

    unet = FreeUUNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="unet", torch_dtype=dtype)
    pipe = CrossImageAttentionStableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", unet=unet, safety_checker=None, torch_dtype=dtype
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
    print("Done.")
    return pipe
