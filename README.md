# Style Transfer with Diffusion Models for Synthetic-to-Real Domain Adaptation

> Estelle Chigot, Dennis G. Wilson, Meriem Ghrib, Thomas Oberlin  
> ISAE-SUPAERO, Airbus
>
> Semantic segmentation models trained on synthetic data often perform poorly on real-world images due to domain gaps, particularly in adverse conditions where labeled data is scarce.
Yet, recent foundation models enable to generate realistic images without any training. This paper proposes to leverage such diffusion models to improve the performance of vision models when learned on synthetic data.
We introduce two novel techniques for semantically consistent style transfer using diffusion models: **C**lass-wise **A**daptive Instance Normalization and **C**ross-atten**TI**on (CACTI) and its extension with selective attention **F**iltering (CACTIF).
CACTI applies statistical normalization selectively based on semantic classes, while CACTIF further filters cross-attention maps based on feature similarity, preventing artifacts in regions with weak cross-attention correspondences.
Our methods transfer style characteristics while preserving semantic boundaries and structural coherence, unlike approaches that apply global transformations or generate content without constraints.
Experiments using GTA5 as source and Cityscapes/ACDC as target domains show that our approach produces higher quality images with lower FID scores and better content preservation.
Our work demonstrates that class-aware diffusion-based style transfer effectively bridges the synthetic-to-real domain gap even with minimal target domain data, advancing robust perception systems for challenging real-world applications.

<a href="https://arxiv.org/abs/2505.16360"><img src="https://img.shields.io/badge/arXiv-2505.16360-b31b1b.svg" height=22.5></a>

![cactif_example](resources/content_style_2605.png)

## `scripts/batch_final.py` – Batch CACTIF Style Transfer (runway-aware)

A command-line script that runs **CACTIF** style transfer over every image in a directory. Each content image is restyled using a reference style image (cross-image attention + AdaIN) together with a text prompt that guides the appearance (e.g. *"Realistic image with fog"*). In this branch, the script can additionally build a **runway mask** from a companion `.txt` annotation and apply runway-specific AdaIN and attention filtering inside it. Results are written to an output directory under the same file names as the inputs, together with a matching runway label file.

---

## Table of contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Quick start](#quick-start)
4. [Command-line arguments](#command-line-arguments)
5. [How style images are chosen](#how-style-images-are-chosen)
6. [Runway masks and annotations](#runway-masks-and-annotations)
7. [What happens per image](#what-happens-per-image)
8. [Output](#output)
9. [Examples](#examples)
10. [Notes and caveats](#notes-and-caveats)
11. [Troubleshooting](#troubleshooting)

---

## Overview

For every file in `--content_dir` that matches `--format`, the script:

1. Picks a style reference image (see [How style images are chosen](#how-style-images-are-chosen)).
2. Inverts both the style and the content image to latents / noise (DDPM inversion), reusing cached latents when available.
3. If `--use_masks` is enabled, builds a runway mask from the content image's `.txt` annotation (and from the style image's, if it has one).
4. Runs the CACTIF diffusion pipeline with cross-image attention and AdaIN callbacks, guided by `--prompt` and `--scale`, with runway-specific AdaIN / attention filtering inside the mask.
5. Center-crops the generated image to a fixed height of **424 px**, saves it to `--output_dir`, and writes a matching runway label `.txt` next to it.

The model is loaded **once** and reused for all images, so large batches are much cheaper than calling the script once per image.

---

## Requirements

- Python 3 with the dependencies of this repository (`torch`, `diffusers`, `numpy`, `Pillow`, `opencv-python`, …).
- **A CUDA-capable GPU.** The script exits immediately if `torch.cuda.is_available()` is `False`.
- The repository modules imported by the script, resolved relative to the project root (the parent of `scripts/`):
  - `cactif_model.CACTIFModel` (must provide `set_runway_mask`, `runway_filter_perc` and `runway_adain_perc`, i.e. the runway-aware branch)
  - `config.Range`, `config.RunConfig`
  - `utils.latent_utils.get_init_latents_and_noises`, `load_or_invert_one_image`
  - `utils.image_utils` (`load_size`)

The script adds the project root to `sys.path` automatically, so it should be launched from the repository root (as in the examples below).

TF32 matmul/cuDNN is enabled and `float32_matmul_precision` is set to `"high"`, which speeds up inference on Ampere or newer GPUs.

---

## Quick start

```bash
python3 scripts/batch_final.py \
    --content_dir /path/to/content_images \
    --format .jpeg \
    --prompt "Realistic image with fog" \
    --scale 0.75 \
    --style_image /path/to/style.jpg \
    --output_dir /path/to/output
```

With the default `--use_masks true`, every content image should have a `.txt` annotation next to it (see [Runway masks and annotations](#runway-masks-and-annotations)); otherwise it is processed without a mask. Pass `--use_masks false` to disable the runway logic.

---

## Command-line arguments

### Required

| Argument | Type | Description |
|---|---|---|
| `--prompt` | str | Prompt controlling the transferred style/content appearance (e.g. `"Realistic image with rain"`). |
| `--scale` | float | CACTIF swap guidance strength. Passed **directly** to the pipeline as `swap_guidance_scale` with no remapping. |
| `--content_dir`, `-content_dir` | path | Directory containing the source (content) images. Not searched recursively. |
| `--format` | str | Image extension to process, e.g. `.jpeg`, `jpeg`, `.png`. A leading dot is added if missing; matching is case-insensitive. Only one extension per run. |
| `--output_dir` | path | Directory that receives the generated images. Created if it does not exist. |

### Style selection (optional)

| Argument | Type | Default | Description |
|---|---|---|---|
| `--style_image` | path | `None` | One reference style image used for **every** content image (cross-image attention and AdaIN). Must be an existing file. |
| `--style_dir` | path | `None` | Directory of style images. A random one is chosen per content image. **Overrides `--style_image`.** Must be an existing directory. |

If neither is given, each content image is used as its own style image (self-style transfer).

### Generation settings (optional)

| Argument | Type | Default | Description |
|---|---|---|---|
| `--name` | str | `CACTIF` | Run name stored in the config (used in generated filenames, e.g. latent caches). |
| `--steps` | int | `50` | Number of diffusion steps. Must be `> 0`. |
| `--skip_steps` | int | `30` | Number of DDPM inversion steps skipped before generation. Must satisfy `0 <= skip_steps < steps`. |
| `--seed` | int | `42` | Random seed. Used for global seeding, the generator, and the random style picker. |
| `--load_latents` | bool | `true` | Reuse cached inverted latents when they exist. |
| `--filtering` | bool | `true` | Enable CACTIF selective attention filtering. |
| `--filter_perc` | float | `0.25` | Fraction of weak attention regions filtered by CACTIF. Must be in `[0, 1]`. |
| `--adain_class` | bool | `false` | Class-wise AdaIN. **Always disabled in this script** – see [Notes](#notes-and-caveats). |

### Runway settings (optional)

| Argument | Type | Default | Description |
|---|---|---|---|
| `--use_masks` | bool | `true` | Build a runway mask from the `.txt` annotation next to each content image and apply runway-specific AdaIN / attention filtering. May be given without a value (means `true`) or as `--use_masks false`. |
| `--runway_filter_perc` | float | `0.5` | Only with `--use_masks`. Fraction of the weakest-similarity tokens **inside the runway** that fall back to content attention. Independent of `--filter_perc`. Must be in `[0, 1]`. |
| `--runway_adain_perc` | float | `0.5` | Only with `--use_masks`. AdaIN strength inside the runway: `1` = full AdaIN, `0` = none. Must be in `[0, 1]`. |

**Boolean values** accept (case-insensitive): `1 true t yes y on` for true and `0 false f no n off` for false.

---

## How style images are chosen

Priority, from highest to lowest:

1. **`--style_dir`**: one random image per content image. Supported extensions: `.png .jpg .jpeg .bmp .webp .tif .tiff`. The random choice uses a `random.Random(seed)` instance, so results are reproducible for a given seed and directory content.
2. **`--style_image`**: the same reference for all content images.
3. **Neither**: each content image acts as its own style reference.

---

## Runway masks and annotations

When `--use_masks` is `true`, the script looks for a text annotation with the **same stem** as each content image (e.g. `img_001.jpeg` → `img_001.txt`, in the same directory).

- Only the **first line** is read. It must contain at least 13 whitespace-separated values.
- The **last 8 values** are the runway corners as coordinates normalized to the original image size, in the order `x_tl y_tl x_bl y_bl x_tr y_tr x_br y_br`. Any leading tokens (e.g. a class id) are kept as-is when the output label is written.
- The polygon is rasterized into a binary mask and cropped to 2:1 the same way the model loads images (`image_utils.load_size`), so it stays aligned with the model input.
- A mask is also built from the **style image's** `.txt`, if one exists; a style image without annotation is simply used without a style mask.
- If a content image has no usable annotation, a warning is printed and that image is processed **without** a mask.

---

## What happens per image

For each content image, `run_style_transfer` does the following inside `torch.inference_mode()`:

1. Updates the latent cache path from the content and style file names.
2. Loads or computes inverted latents and noise for the **style** image, then for the **content** image.
3. Sets both on the model and builds the initial latents/noises for generation.
4. Resets the runway mask on the model (so a mask never leaks between images). If `--use_masks` is on and an annotation exists, builds the content (and optional style) runway mask and passes it to the model at latent resolution.
5. Computes the cross-image attention step range as the union of the 32×32 and 64×64 attention ranges from the config.
6. Runs the pipeline with `guidance_scale=1.0`, `eta=1`, the AdaIN callback, `swap_guidance_scale=--scale`, and a CUDA generator seeded with `--seed`. The prompt is replicated three times (one per latent in the batch: style, content, and the generated result). The runway mask drives the runway-specific AdaIN (`--runway_adain_perc`) and attention filtering (`--runway_filter_perc`).
7. Takes the first image returned by the pipeline, center-crops its height to `OUTPUT_TARGET_HEIGHT = 424`, and saves it. The diffusion itself runs at full resolution, so the crop does not affect the mask.
8. If a mask was used, transforms the original runway corners into the cropped output's coordinate space and writes them to a `.txt` label next to the output image.

Because latents are cached per content/style pair, re-running the same pair with `--load_latents true` skips the expensive inversion.

---

## Output

- Files are saved in `--output_dir` with the **same file name** as the input (e.g. `img_001.jpeg` → `<output_dir>/img_001.jpeg`). Existing files with the same name are overwritten.
- The pipeline generates images 512 px tall; the script **center-crops the height to 424 px** (split evenly top/bottom). The width is unchanged. To change this, edit `OUTPUT_TARGET_HEIGHT` at the top of the script.
- For each image processed **with a mask**, a label file `<output_dir>/img_001.txt` is written: same format and leading tokens as the source annotation, with the 8 corner values re-normalized to the cropped output image. No label is written when the image had no annotation or `--use_masks false`.
- Progress is printed per image, e.g.:

```
[3/120] img_003.jpeg style=uda_sample_fog.jpg -> /…/fog/img_003.jpeg
```

---

## Examples

The examples below share these settings:

```bash
scale=0.75
content_dir="/home/jahn_pa/projects/ges"
runway_filter_perc=1
runway_adain_perc=0
```


```bash
prompt="Realistic image with rain"
style_image="/home/jahn_pa/projects/weather_samples/rain/pexels-splitshire-1553.jpg"
out="/home/jahn_pa/projects/cactif_output/rain"

python3 scripts/batch_final.py \
      --content_dir "$content_dir" \
      --format .jpeg \
      --prompt "$prompt" \
      --scale "$scale" \
      --runway_filter_perc "$runway_filter_perc" \
      --runway_adain_perc "$runway_adain_perc" \
      --style_image "$style_image" \
      --output_dir "$out"
```

### Random style per image

```bash
python3 scripts/batch_final.py \
      --content_dir "$content_dir" \
      --format .jpeg \
      --prompt "Realistic image with rain" \
      --scale 0.75 \
      --runway_filter_perc "$runway_filter_perc" \
      --runway_adain_perc "$runway_adain_perc" \
      --style_dir "/home/jahn_pa/projects/weather_samples/rain" \
      --output_dir "/home/jahn_pa/projects/cactif_output/rain_random"
```

### Self-style transfer (no style image)

```bash
python3 scripts/batch_final.py \
      --content_dir "$content_dir" \
      --format .png \
      --prompt "Realistic image with snow" \
      --scale 0.75 \
      --runway_filter_perc "$runway_filter_perc" \
      --runway_adain_perc "$runway_adain_perc" \
      --output_dir "/home/jahn_pa/projects/cactif_output/self"
```

### Without runway masks

```bash
python3 scripts/batch_final.py \
      --content_dir "$content_dir" \
      --format .jpeg \
      --prompt "Realistic image with rain" \
      --scale 0.75 \
      --use_masks false \
      --style_image "$style_image" \
      --output_dir "/home/jahn_pa/projects/cactif_output/rain_nomask"
```

### Full set of options

```bash
python3 scripts/batch_final.py \
      --content_dir "$content_dir" \
      --format jpeg \
      --prompt "Realistic image with fog" \
      --scale 0.75 \
      --style_image "$style_image" \
      --output_dir "$out" \
      --name CACTIF \
      --steps 50 \
      --skip_steps 30 \
      --seed 42 \
      --load_latents true \
      --filtering true \
      --filter_perc 0.25 \
      --adain_class false \
      --use_masks true \
      --runway_filter_perc 0.5 \
      --runway_adain_perc 0.5
```

---

## Notes and caveats

- **`--adain_class` is always off.** Class-wise AdaIN needs semantic labels, which this image-only CLI does not provide. If you set it to `true`, a warning is printed and the config is still created with `adain_class=False`. Runway AdaIN is controlled separately by `--runway_adain_perc`.
- **`--scale` is not remapped.** The value is used as-is for `swap_guidance_scale`.
- **`--filter_perc` vs `--runway_filter_perc`.** The former applies to the whole image (and only when `--filtering` is `true`); the latter applies only inside the runway mask and is independent of it.
- **`--use_masks` defaults to `true`.** Images without a `.txt` annotation are still processed, but without a mask and without an output label (a warning is printed).
- **`--style_dir` wins over `--style_image`** when both are supplied.
- **Single extension per run.** To process mixed formats, run the script once per extension (with different or the same `--output_dir`).
- **Same-name outputs.** Two inputs with the same name but different extensions (e.g. `a.jpeg`, `a.png`) are only an issue if you reuse one output directory across runs, since file names (and label `.txt` stems) are preserved.
- **Latent cache.** Cached latents are keyed by the content and style file stems. If you change the images behind a name, or settings such as `--steps` / `--skip_steps`, run with `--load_latents false` to force re-inversion.
- **Helper functions not exposed on the CLI.** `fix_bad_labels` (rewrites output labels from the original annotations) and `save_output_overlay` (label sanity-check overlay; its call is commented out in `run_style_transfer`) exist in the script but are not reachable from the command line.

---

## Troubleshooting

| Message / symptom | Cause and fix |
|---|---|
| `CACTIF inference requires CUDA in this repository.` | No CUDA GPU visible to PyTorch. Check drivers and `CUDA_VISIBLE_DEVICES`. |
| `No files with format '.jpeg' found in …` | No file in `--content_dir` has that extension (check spelling and that files are not in a subfolder). |
| `Input directory does not exist: …` | `--content_dir` path is wrong. |
| `No style images found in …` | `--style_dir` contains none of the supported extensions. |
| `Warning: no usable .txt annotation for … running without mask.` | Missing/empty `.txt` next to the content image, or fewer than 13 values on its first line. Add the annotation or pass `--use_masks false`. |
| `--runway_filter_perc must be in [0, 1].` / `--runway_adain_perc must be in [0, 1].` | Value out of range. |
| `--skip_steps must be >= 0 and < --steps.` | Lower `--skip_steps` or raise `--steps`. |
| `AttributeError: … set_runway_mask` | You are on a branch whose `CACTIFModel` lacks the runway-aware code. |
| `ModuleNotFoundError: cactif_model` / `cv2` | Run from the repository root with the repo layout intact, and install `opencv-python`. |

---