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

## Description  
Official implementation of Style Transfer with Diffusion Models for Synthetic-to-Real Domain Adaptation (CACTIF).


## Environment
Our code builds on the requirement of the `diffusers` library. The current environment target is `python 3.11.6`.
To set up the CACTIF environment, please run:
```
conda env create -f environment/environment.yaml
conda activate cactif
```

Note:
- `xformers` is not required by this repository setup.
- Attention uses PyTorch scaled dot product attention backends.

Please download the [GTA5 dataset](https://download.visinf.tu-darmstadt.de/data/from_games/), and link it to the `data` folder of the repository:
```
ln -s /path/to/gta5/ /path/to/CACTIF/data/gta
```

In `data/style` you should put any style image in Cityscapes or ACDC naming convention and their corresponding label. We put a few examples in this repository.

In the end you data folder should look like this:
```
DAFormer
├── ...
├── data
│   ├── gta
│   │   ├── images
│   │   ├── labels
│   ├── style
│   │   ├── images
│   │   ├── labels
├── ...

```

To use the Rare Class Sampling (RCS) strategy during dataset generation, you can follow the [DAFormer data preprocessing](https://github.com/lhoyer/DAFormer) steps.


## Usage  

To generate an image, you can simply run the `run.py` script. For example,
```
# Run CACTIF
python run.py
# Run CACTI
python run.py --name CACTI --filtering False
```
Notes:
- You can specify a name with `--name your_name`.
- You can switch off the filtering operation with `--filtering false`, and the class AdaIN module with `--adain_class false`.
- You can specify the number of images to generate for each style image with `--nb_img_per_style n`.
- You can set `--load_latents` to `True` to load the latents from a file instead of inverting the input images every time. 
  - This is useful if you want to generate multiple images with the same structure but different appearances.


## Batch Inference Script (New)

For folder-based generation with prompt guidance and style references, use:
```
python scripts/batch_canny_depth_control.py \
  --prompt "rainy evening urban street" \
  --scale 0.7 \
  --content_dir data/gta/images \
  --format .png \
  --output_dir output/batch \
  --style_image data/style/images/example_leftImg8bit.png
```

Optional variability mode (overrides `--style_image`):
```
python scripts/batch_canny_depth_control.py \
  --prompt "rainy evening urban street" \
  --scale 0.7 \
  --content_dir data/gta/images \
  --format .png \
  --output_dir output/batch \
  --style_dir data/style/images
```

### CLI Behavior
- The script scans `--content_dir` and selects all files matching `--format`.
- Output filenames are preserved: each generated file is written to `--output_dir/<input_name>`.
- `--prompt` is used as additional semantic guidance during inversion and generation.
- `--style_image` sets one global style reference for all content images.
- `--style_dir` selects one random style image per content image and takes precedence over `--style_image`.
- If neither `--style_image` nor `--style_dir` is provided, the script uses self-style transfer (each content image as its own style reference).


## End-to-End Pipeline Logic

This repository now exposes two practical inference entry points:
- `run.py`: dataset-oriented CACTIF/CACTI transfer using content labels and style labels.
- `scripts/batch_canny_depth_control.py`: image-folder batch transfer driven by prompt + style image(s).

In both cases, the core model path is CACTIF with cross-image attention editing.

### Step-by-step flow in `scripts/batch_canny_depth_control.py`
1. Parse CLI arguments and validate ranges (`--scale`, `--steps`, `--skip_steps`, etc.).
2. Build a `RunConfig` object that carries diffusion and editing settings.
3. Load CACTIF model wrapper (`CACTIFModel`) and set scheduler timesteps.
4. For each content image:
   - Choose style image (`--style_dir` random sample, else `--style_image`, else self-style).
   - Invert style image to latent/noise (or load cached latents if available).
   - Invert content image to latent/noise (or load cached latents if available).
   - Build triplet latents `[transfer, style, content]` and corresponding noise schedules.
   - Run CACTIF denoising with prompt guidance and selective attention filtering.
   - Save only the transferred output image to `--output_dir` with original filename.


## Models and Checkpoints Loaded

The full loading process is centered on the Stable Diffusion v1.5 checkpoint and custom CACTIF modifications.

### 1) Base diffusion checkpoint
- Checkpoint: `runwayml/stable-diffusion-v1-5`
- Loaded by `CrossImageAttentionStableDiffusionPipeline.from_pretrained(...)`
- Provides the base components (text encoder, tokenizer, VAE, UNet scaffold, scheduler config).

### 2) Custom UNet replacement
- Checkpoint source: `runwayml/stable-diffusion-v1-5`, subfolder `unet`
- Class: `FreeUUNet2DConditionModel`
- The default UNet from the pipeline is replaced with this CACTIF-compatible UNet implementation.

### 3) Scheduler selection
- Checkpoint source: `runwayml/stable-diffusion-v1-5`, subfolder `scheduler`
- Class: `DDIMScheduler`
- Used for both inversion and forward generation timesteps.

### 4) Inversion stage (per style/content image)
- Function path: `utils/latent_utils.py` -> `utils/ddpm_inversion.py`
- The VAE encodes each image into latent space.
- DDPM inversion computes:
  - latent trajectory (`wts`)
  - noise maps (`zs`)
- Artifacts can be cached under `latents/latents_<num_steps>/(style|content)` and reused when `--load_latents true`.

### 5) CACTIF editing/generation stage
- Forward call: `model.pipe(...)` through `CACTIFModel`
- Active mechanisms:
  - Cross-image attention transfer (content/layout + style appearance coupling)
  - Optional CACTIF filtering (`--filtering`, `--filter_perc`)
  - AdaIN callback path (class AdaIN disabled in image-only batch script by default)
- Prompt integration:
  - The same prompt is passed to the triplet batch entries.
  - `--scale` maps to `swap_guidance_scale`, controlling prompt/style influence strength.


## Practical Notes on Runtime

- You typically see three progress phases per output image:
  - style inversion
  - content inversion
  - denoising/generation
- With `--load_latents true`, repeated style/content pairs skip inversion and run faster.
- The generation phase is usually the longest because it applies cross-image attention edits and optional filtering inside attention layers.


## Acknowledgements 
This code uses a lot of resources from the [Cross-image attention](https://github.com/garibida/cross-image-attention) repository. We sincerely thank the authors for sharing their work.


## Citation
If you use this code for your research, please cite the following work: 
```
@article{Chigot_2025,
   title={Style transfer with diffusion models for synthetic-to-real domain adaptation},
   journal={Computer Vision and Image Understanding},
   volume={259},
   pages={104445}
   ISSN={1077-3142},
   DOI={10.1016/j.cviu.2025.104445},
   author={Chigot, Estelle and Wilson, Dennis G. and Ghrib, Meriem and Oberlin, Thomas},
   year={2025},
}
```
