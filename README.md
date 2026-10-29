# Anviksha Generator — 64×64 image and video synthesis

A small generative model trained from scratch, packaged as a Hugging Face Space
and as a GitHub Actions training job. It generates images, and it generates
video by walking between two points in latent space.

**Read the limits before the features.** At this scale the output is not
photorealistic, and the video is not a learned motion model. Both points are
explained below rather than hidden.

## What it actually is

| Component | Detail |
|---|---|
| Model | DCGAN, 64×64 RGB |
| Latent | 128-d |
| Generator | ~1.2 M parameters at `base=32` |
| Discriminator | ~1.1 M parameters |
| Training data | 1,542 images (Indian dance dataset, Apache-2.0) |
| Training time | ~44 minutes for 40 epochs on 2 CPU cores |
| Output size | 64×64 PNG, 48-frame MP4 |

Measured on 2 CPU cores: 0.89 s per step at batch 16, 66 s per epoch.

## Honest expectations

**Images.** A DCGAN with 1.2 M parameters trained on 1,542 images for an hour
produces 64×64 images that capture palette and large structure — the blue of a
pool, a warm skin tone, a dark stage background — and lose faces, hands and
detail. This is a working pipeline, not a photorealistic generator. Reaching
that standard needs a diffusion model, hundreds of thousands of images and a
GPU.

**Video.** The video tab interpolates between two latent vectors using spherical
interpolation and decodes each step. It produces smooth motion, but it is
**motion in latent space, not a temporal model**. There is no understanding of
what should happen next, no physics, no continuity of identity across frames
beyond what the generator's latent space happens to preserve. A real video
generator needs video training data and far more compute.

## Project layout

```
app.py                      Gradio UI: image tab + video tab
src/models.py               Generator, Discriminator, latent walk, video writer
src/train_gan.py            DCGAN training loop, resumable
src/download_images.py      fetch the training images from the Hugging Face Hub
configs/default.json        exact hyperparameters
.github/workflows/train.yml cloud-CPU training job
```

## Run the UI locally

```bash
pip install -r requirements.txt
python src/download_images.py --out data/raw
python src/train_gan.py --data data/images --epochs 40 --out weights
python app.py
```

## Train on cloud CPU

### Option A — Hugging Face Space

1. Create a Space: <https://huggingface.co/new-space> → SDK **Gradio** →
   hardware **CPU basic (2 vCPU, 16 GB)**.
2. Upload this repository to the Space.
3. Add a `weights/gan.pt` checkpoint (train one first, or run the training job
   below and upload its artifact).
4. The Space serves the UI. Without a checkpoint the app says so instead of
   pretending to generate.

To train *inside* the Space, add a `train.py` entry that calls
`src/train_gan.py` with `--time-budget`, and enable persistent storage so the
checkpoint survives restarts. Free Spaces sleep after inactivity and lose
non-persistent files.

### Option B — GitHub Actions (2 vCPU, 7 GB, free minutes)

`.github/workflows/train.yml` runs on a public-repo runner. Trigger it from the
Actions tab or with:

```bash
gh workflow run train-gan -f epochs=12 -f minutes=50
```

It downloads the images, trains within the time budget, and uploads
`weights/gan.pt` plus per-epoch sample grids as an artifact. Runners are capped
at 6 hours per job, so long training is done as repeated runs: the checkpoint is
resumable and `--resume` continues from the last epoch.

## Why the training recipe looks like this

- **`betas=(0.5, 0.999)`** — the standard DCGAN setting. The default 0.9 makes
  the discriminator overpower the generator in the first epochs.
- **One D step per G step, equal learning rates** — simple and adequate at this
  data size. If the discriminator runs away (its loss drops below ~0.05 while
  the generator loss climbs past ~5), halve the discriminator learning rate.
- **Fixed noise for sample grids** — the per-epoch PNGs are comparable across
  epochs; the same 16 latent vectors are decoded every time.
- **Resumable checkpoints** — G, D, both optimisers and the epoch counter are
  saved together, so a stopped run continues without a loss spike.

## Licence

Code: Apache-2.0. Training images: `tanmaykm/indian_dance_forms`, Apache-2.0,
downloaded at runtime and not redistributed here.
