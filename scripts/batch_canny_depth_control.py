import argparse
from pathlib import Path
import random
import sys

import torch
from diffusers.training_utils import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cactif_model import CACTIFModel
from config import Range, RunConfig
from utils.latent_utils import get_init_latents_and_noises, load_or_invert_one_image
import os
import cv2
import numpy as np
from PIL import Image


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def normalize_format(fmt: str) -> str:
    fmt = fmt.strip().lower()
    if not fmt.startswith("."):
        fmt = "." + fmt
    return fmt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch CACTIF style transfer with a style reference image and prompt guidance."
    )

    parser.add_argument(
        "--prompt",
        required=True,
        help=(
            "Prompt controlling the transferred "
            "style/content appearance."
        ),
    )

    parser.add_argument(
        "--scale",
        type=float,
        required=True,
        help=(
            "Direct CACTIF swap guidance strength. "
            "This value is passed as swap_guidance_scale without remapping."
        ),
    )

    parser.add_argument(
        "-content_dir",
        "--content_dir",
        required=True,
        type=Path,
        help="Directory containing source images.",
    )

    parser.add_argument(
        "--format",
        required=True,
        help="Image extension to process, e.g. .jpeg, jpeg, .png.",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Directory receiving the generated images.",
    )

    parser.add_argument(
        "--style_image",
        type=Path,
        default=None,
        help=(
            "Reference style image used by IP-Adapter. "
            "If omitted, each input image is used as its own "
            "style image (self-style transfer)."
        ),
    )

    parser.add_argument(
        "--style_dir",
        type=Path,
        default=None,
        help=(
            "Optional directory of style images. If provided, one random style image "
            "is selected for each input image and this overrides --style_image."
        ),
    )

    parser.add_argument(
        "--name",
        type=str,
        default="CACTIF",
        help="Run name used in generated filenames.",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of diffusion steps (default: 50).",
    )

    parser.add_argument(
        "--skip_steps",
        type=int,
        default=30,
        help="Number of DDPM inversion steps to skip before generation (default: 30).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )

    parser.add_argument(
        "--load_latents",
        type=parse_bool,
        default=True,
        help="Load cached latents when available (true/false).",
    )

    parser.add_argument(
        "--filtering",
        type=parse_bool,
        default=True,
        help="Enable CACTIF selective attention filtering (default: true).",
    )

    parser.add_argument(
        "--filter_perc",
        type=float,
        default=0.25,
        help="Percentage of weak attention regions filtered by CACTIF (default: 0.25).",
    )

    parser.add_argument(
        "--adain_class",
        type=parse_bool,
        default=False,
        help=(
            "Enable class-wise AdaIN. Defaults to false in this script because only image paths "
            "are provided by CLI (no semantic labels)."
        ),
    )

    parser.add_argument(
        "--use_masks",
        type=parse_bool,
        nargs="?",
        const=True,
        default=True,
        help="Build a runway mask from the .txt annotation next to each content image "
             "and apply runway-specific AdaIN / attention filtering.",
    )

    parser.add_argument(
        "--runway_filter_perc",
        type=float,
        default=0.5,
        help="Only with --use_masks. Fraction of weakest-similarity tokens inside the runway "
             "that fall back to content attention (independent of --filter_perc).",
    )

    parser.add_argument(
        "--runway_adain_perc",
        type=float,
        default=0.5,
        help="Only with --use_masks. AdaIN strength inside the runway: 1 = full AdaIN, 0 = none.",
    )

    args = parser.parse_args()

    if args.steps <= 0:
        parser.error("--steps must be > 0.")

    if args.skip_steps < 0 or args.skip_steps >= args.steps:
        parser.error("--skip_steps must be >= 0 and < --steps.")

    if not 0.0 <= args.filter_perc <= 1.0:
        parser.error("--filter_perc must be in [0, 1].")

    if args.style_image is not None and not args.style_image.is_file():
        parser.error(f"Style image does not exist: {args.style_image}")

    if args.style_dir is not None and not args.style_dir.is_dir():
        parser.error(f"Style directory does not exist: {args.style_dir}")

    if not 0.0 <= args.runway_filter_perc <= 1.0:
        parser.error("--runway_filter_perc must be in [0, 1].")

    if not 0.0 <= args.runway_adain_perc <= 1.0:
        parser.error("--runway_adain_perc must be in [0, 1].")

    return args



def load_gt_points_from_txt(img_path, w, h):
    """Read a polygon annotation (last 8 values = normalized corner coords)."""
    txt_path = os.path.splitext(img_path)[0] + ".txt"

    if not os.path.exists(txt_path):
        return None

    with open(txt_path, "r", encoding="utf-8") as f:
        line = f.readline().strip()

    if not line:
        return None

    values = line.split()
    if len(values) < 13:
        return None

    coords = list(map(float, values[-8:]))

    x_tl, y_tl, x_bl, y_bl, x_tr, y_tr, x_br, y_br = coords

    gt_pts = np.array(
        [
            [x_tl * w, y_tl * h],
            [x_tr * w, y_tr * h],
            [x_br * w, y_br * h],
            [x_bl * w, y_bl * h],
        ],
        dtype=np.int32,
    ).reshape(-1, 1, 2)

    return gt_pts

def load_gt_mask(img_path, w, h):
    """Build a binary runway mask (H, W) float32 in {0, 1} from the GT polygon.

    Returns None if no companion .txt annotation exists for this image, so
    callers can fall back to skipping the runway re-integration step.
    """
    gt_pts = load_gt_points_from_txt(img_path, w, h)
    if gt_pts is None:
        return None

    gt_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(gt_mask, [gt_pts], 255)

    return (gt_mask > 0).astype(np.float32)



def build_runway_mask(content_img: Path, crop_square: bool = True):
    """Runway mask at original-image resolution, center-cropped like image_utils.load_size."""
    with Image.open(content_img) as im:
        w, h = im.size
    mask = load_gt_mask(str(content_img), w, h)
    if mask is None:
        return None
    if crop_square:
        if h < w:
            off = (w - h) // 2
            mask = mask[:, off:off + h]
        elif w < h:
            off = (h - w) // 2
            mask = mask[off:off + w, :]
    return mask



def collect_content_images(content_dir: Path, image_format: str):
    input_files = sorted(
        path
        for path in content_dir.iterdir()
        if path.is_file() and path.suffix.lower() == image_format
    )

    if not input_files:
        raise SystemExit(f"No files with format '{image_format}' found in {content_dir}")

    return input_files


def collect_style_images(style_dir: Path):
    allowed_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    style_files = sorted(
        path
        for path in style_dir.iterdir()
        if path.is_file() and path.suffix.lower() in allowed_suffixes
    )

    if not style_files:
        raise SystemExit(f"No style images found in {style_dir}")

    return style_files


def run_style_transfer(
    model: CACTIFModel,
    cfg: RunConfig,
    content_img: Path,
    style_img: Path,
    output_path: Path,
    use_masks: bool = False,
) -> None:
    with torch.inference_mode():
        cfg.update_latents_path(content_img.stem, style_img.stem)
        latents_style, noise_style = load_or_invert_one_image(model.pipe, cfg, img_path=style_img, type_img="style")
        latents_content, noise_content = load_or_invert_one_image(model.pipe, cfg, img_path=content_img, type_img="content")

        model.set_latents(latents_style, latents_content)
        model.set_noise(noise_style, noise_content)

        # Runway mask (content image only). Always reset so a mask never leaks between images.
        model.set_runway_mask(None)  # always reset so a mask never leaks between images
        if use_masks:
            mask = build_runway_mask(content_img, crop_square=True)
            if mask is None:
                print(f"  Warning: no usable .txt annotation for {content_img.name}; running without mask.")
            else:
                ref = latents_content[0] if isinstance(latents_content, (list, tuple)) else latents_content
                model.set_runway_mask(mask, latent_hw=ref.shape[-2:], device=ref.device)

        init_latents, init_zs = get_init_latents_and_noises(model=model, cfg=cfg)
        start_step = min(cfg.cross_attn_32_range.start, cfg.cross_attn_64_range.start)
        end_step = max(cfg.cross_attn_32_range.end, cfg.cross_attn_64_range.end)

        model.enable_edit = True
        result = model.pipe(
            prompt=[cfg.prompt] * 3,
            latents=init_latents,
            guidance_scale=1.0,
            num_inference_steps=cfg.num_timesteps,
            swap_guidance_scale=cfg.swap_guidance_scale,
            callback=model.get_adain_callback(),
            eta=1,
            zs=init_zs,
            generator=torch.Generator("cuda").manual_seed(cfg.seed),
            cross_image_attention_range=Range(start=start_step, end=end_step),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(output_path)


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CACTIF inference requires CUDA in this repository.")

    # TF32 tensor cores (Ampere or newer GPUs)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    content_dir = args.content_dir
    output_dir = args.output_dir
    image_format = normalize_format(args.format)

    if not content_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {content_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    input_files = collect_content_images(content_dir, image_format)
    style_files = collect_style_images(args.style_dir) if args.style_dir is not None else None

    if args.adain_class:
        print("Warning: --adain_class requires semantic labels; disabling it for this image-only CLI flow.")

    cfg = RunConfig(
        prompt=args.prompt,
        seed=args.seed,
        num_timesteps=args.steps,
        skip_steps=args.skip_steps,
        load_latents=args.load_latents,
        output_path=output_dir,
        name=args.name,
        filtering=args.filtering,
        adain_class=False,
        filter_perc=args.filter_perc,
        swap_guidance_scale=args.scale,
    )

    set_seed(cfg.seed)
    model = CACTIFModel(cfg)
    model.runway_filter_perc = args.runway_filter_perc
    model.runway_adain_perc = args.runway_adain_perc

    
    model.pipe.scheduler.set_timesteps(cfg.num_timesteps)

    if style_files is not None:
        print(f"Using random style reference per image from: {args.style_dir}")
        print(f"Discovered {len(style_files)} style file(s).")
    elif args.style_image is None:
        print("No --style_image provided: using each input image as its own style reference.")
    else:
        print(f"Using global style reference: {args.style_image}")

    print(f"Found {len(input_files)} input file(s).")
    print(f"Output folder: {output_dir}")

    rng = random.Random(cfg.seed)
    for index, input_path in enumerate(input_files, start=1):
        output_path = output_dir / input_path.name
        if style_files is not None:
            style_path = rng.choice(style_files)
        else:
            style_path = args.style_image if args.style_image is not None else input_path

        print(
            f"[{index}/{len(input_files)}] "
            f"{input_path.name} style={style_path.name} -> {output_path}"
        )

        run_style_transfer(
            model=model,
            cfg=cfg,
            content_img=input_path,
            style_img=style_path,
            output_path=output_path,
        )

    print(f"Finished. Wrote {len(input_files)} file(s) to {output_dir}")


if __name__ == "__main__":
    main()