import argparse
from pathlib import Path
from typing import List, Tuple

import torch
from diffusers.training_utils import set_seed

from cactif_model import CACTIFModel
from config import Range, RunConfig
import palettes
from utils.latent_utils import (
    get_init_latents_and_noises,
    load_or_invert_one_image,
)
from utils.mask_utils import process_label


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def normalize_format(fmt: str) -> str:
    image_format = fmt.strip().lower()
    if not image_format.startswith("."):
        image_format = "." + image_format
    return image_format


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch CACTIF style transfer using content and style image/label pairs."
    )

    parser.add_argument(
        "--prompt",
        required=True,
        help="Prompt used by CACTIF during inversion and generation.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        required=True,
        help="Prompt/style transfer strength in [0, 1], mapped to CACTIF swap guidance.",
    )
    parser.add_argument(
        "-content_dir",
        "--content_dir",
        required=True,
        type=Path,
        help="Directory containing source content images (typically data/.../images).",
    )
    parser.add_argument(
        "--content_label_dir",
        type=Path,
        default=None,
        help="Directory containing source content labels. Defaults to sibling labels folder.",
    )
    parser.add_argument(
        "--format",
        required=True,
        help="Image extension to process, e.g. .png or png.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Directory receiving generated images.",
    )
    parser.add_argument(
        "--style_image",
        type=Path,
        default=None,
        help="Optional single style image path. If omitted, styles are discovered in --style_dir.",
    )
    parser.add_argument(
        "--style_label",
        type=Path,
        default=None,
        help="Label path for --style_image. Required when --style_image is used.",
    )
    parser.add_argument(
        "--style_dir",
        type=Path,
        default=Path("data/style"),
        help="Root style directory containing images/ and labels/ subfolders.",
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
        help="Number of diffusion steps.",
    )
    parser.add_argument(
        "--skip_steps",
        type=int,
        default=30,
        help="Number of DDPM inversion steps to skip before generation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
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
        help="Enable CACTIF selective attention filtering.",
    )
    parser.add_argument(
        "--adain_class",
        type=parse_bool,
        default=True,
        help="Enable class-wise AdaIN.",
    )
    parser.add_argument(
        "--filter_perc",
        type=float,
        default=0.25,
        help="Percentage of weak attention regions filtered by CACTIF.",
    )
    parser.add_argument(
        "--nb_img_per_style",
        type=int,
        default=None,
        help="Optional cap for number of content images per style reference.",
    )

    args = parser.parse_args()

    if not 0.0 <= args.scale <= 1.0:
        parser.error("--scale must be between 0 and 1.")
    if args.steps <= 0:
        parser.error("--steps must be > 0.")
    if args.skip_steps < 0 or args.skip_steps >= args.steps:
        parser.error("--skip_steps must be >= 0 and < --steps.")
    if args.filter_perc < 0 or args.filter_perc > 1:
        parser.error("--filter_perc must be in [0, 1].")
    if args.nb_img_per_style is not None and args.nb_img_per_style <= 0:
        parser.error("--nb_img_per_style must be > 0 when provided.")

    if args.style_image is not None:
        if not args.style_image.is_file():
            parser.error(f"Style image does not exist: {args.style_image}")
        if args.style_label is None:
            parser.error("--style_label is required when --style_image is set.")
        if not args.style_label.is_file():
            parser.error(f"Style label does not exist: {args.style_label}")

    return args


def resolve_content_label_dir(content_dir: Path, explicit_label_dir: Path = None) -> Path:
    if explicit_label_dir is not None:
        return explicit_label_dir

    if content_dir.name.lower() == "images":
        return content_dir.parent / "labels"

    return content_dir / "labels"


def collect_content_pairs(content_dir: Path, content_label_dir: Path, image_format: str) -> List[Tuple[Path, Path]]:
    content_images = sorted(
        path
        for path in content_dir.iterdir()
        if path.is_file() and path.suffix.lower() == image_format
    )

    if not content_images:
        raise SystemExit(f"No files with format '{image_format}' found in {content_dir}")

    pairs: List[Tuple[Path, Path]] = []
    missing_labels = []
    for image_path in content_images:
        label_path = content_label_dir / image_path.name
        if not label_path.is_file():
            missing_labels.append(label_path)
            continue
        pairs.append((image_path, label_path))

    if missing_labels:
        preview = "\n".join(str(path) for path in missing_labels[:5])
        raise SystemExit(
            "Missing content labels for one or more input images. "
            "Expected matching filenames in content label directory. "
            f"Examples:\n{preview}"
        )

    return pairs


def collect_style_refs(args: argparse.Namespace) -> List[Tuple[str, Path, Path]]:
    if args.style_image is not None:
        style_name = args.style_image.stem
        return [(style_name, args.style_image, args.style_label)]

    style_image_root = args.style_dir / "images"
    style_label_root = args.style_dir / "labels"
    if not style_image_root.is_dir() or not style_label_root.is_dir():
        raise SystemExit(
            "Style directory must contain images/ and labels/ subfolders. "
            f"Got: {args.style_dir}"
        )

    style_refs: List[Tuple[str, Path, Path]] = []
    for style_label in sorted(style_label_root.iterdir()):
        if not style_label.is_file() or "olor.png" not in style_label.name.split("_")[-1]:
            continue

        style_name = style_label.name.split("_gt")[0]
        left_img = style_image_root / f"{style_name}_leftImg8bit.png"
        anon_img = style_image_root / f"{style_name}_rgb_anon.png"
        style_image = left_img if left_img.is_file() else anon_img
        if not style_image.is_file():
            continue

        style_refs.append((style_name, style_image, style_label))

    if not style_refs:
        raise SystemExit(
            "No valid style references found. Expected style labels and matching style images in data/style."
        )

    return style_refs


def run_style_transfer(
    model: CACTIFModel,
    cfg: RunConfig,
    content_img: Path,
    content_label: Path,
    style_img: Path,
    style_label: Path,
    output_path: Path,
) -> None:
    with torch.no_grad():
        label_style, label_style_adain = process_label(style_label, palettes.CITYSCAPES)
        model.label_style = [label_style]
        model.label_style_adain = label_style_adain
        model.label_content, model.label_content_adain = process_label(content_label, palettes.GTA)

        cfg.update_latents_path(content_img.stem, style_img.stem)
        latents_style, noise_style = load_or_invert_one_image(model.pipe, cfg, img_path=style_img, type_img="style")
        latents_content, noise_content = load_or_invert_one_image(
            model.pipe, cfg, img_path=content_img, type_img="content"
        )

        model.set_latents(latents_style, latents_content)
        model.set_noise(noise_style, noise_content)
        model.set_onehot_masks()

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

    image_format = normalize_format(args.format)
    content_dir = args.content_dir
    output_dir = args.output_dir

    if not content_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {content_dir}")

    content_label_dir = resolve_content_label_dir(content_dir, args.content_label_dir)
    if not content_label_dir.is_dir():
        raise SystemExit(f"Content label directory does not exist: {content_label_dir}")

    content_pairs = collect_content_pairs(content_dir, content_label_dir, image_format)
    style_refs = collect_style_refs(args)

    if args.nb_img_per_style is not None:
        content_pairs = content_pairs[: args.nb_img_per_style]

    cfg = RunConfig(
        prompt=args.prompt,
        seed=args.seed,
        num_timesteps=args.steps,
        skip_steps=args.skip_steps,
        load_latents=args.load_latents,
        output_path=output_dir,
        name=args.name,
        filtering=args.filtering,
        adain_class=args.adain_class,
        filter_perc=args.filter_perc,
        swap_guidance_scale=args.scale,
    )

    set_seed(cfg.seed)
    model = CACTIFModel(cfg)
    model.pipe.scheduler.set_timesteps(cfg.num_timesteps)

    print(f"Found {len(content_pairs)} content image(s).")
    print(f"Found {len(style_refs)} style reference(s).")
    print(f"Content image folder: {content_dir}")
    print(f"Content label folder: {content_label_dir}")
    print(f"Output folder: {output_dir}")

    total_jobs = len(content_pairs) * len(style_refs)
    current_job = 0
    for style_name, style_img, style_label in style_refs:
        for content_img, content_label in content_pairs:
            current_job += 1

            if len(style_refs) == 1:
                output_path = output_dir / content_img.name
            else:
                output_path = output_dir / style_name / content_img.name

            print(
                f"[{current_job}/{total_jobs}] "
                f"content={content_img.name} style={style_name} -> {output_path}"
            )

            run_style_transfer(
                model=model,
                cfg=cfg,
                content_img=content_img,
                content_label=content_label,
                style_img=style_img,
                style_label=style_label,
                output_path=output_path,
            )

    print(f"Finished. Wrote {total_jobs} file(s) to {output_dir}")


if __name__ == "__main__":
    main()