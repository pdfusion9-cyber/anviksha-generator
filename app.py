"""Gradio interface for the generator — image synthesis and latent-walk video.

Deploy as a Hugging Face Space (SDK: gradio). The Space loads `weights/gan.pt`
if it exists; without a checkpoint the app says so instead of pretending to
generate.

Run locally:
    python app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from models import Generator, latent_walk_frames, write_video  # noqa: E402

WEIGHTS = Path(os.environ.get("GAN_WEIGHTS", "weights/gan.pt"))
SAMPLE_DIR = Path("samples")

_state: dict = {"generator": None, "meta": {}}


def load_generator() -> tuple[Generator | None, str]:
    if _state["generator"] is not None:
        return _state["generator"], _state["meta"].get("status", "")

    if not WEIGHTS.exists():
        return None, (
            "No checkpoint found at "
            f"`{WEIGHTS}`. Train one first with `python src/train_gan.py`, "
            "or upload a `gan.pt` produced by that script."
        )

    checkpoint = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    generator = Generator(
        z_dim=checkpoint.get("z_dim", 128),
        base=checkpoint.get("base", 32),
        out_size=checkpoint.get("size", 64),
    )
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    _state["generator"] = generator
    _state["meta"] = {
        "status": f"loaded checkpoint from epoch {checkpoint.get('epoch', '?')}",
        "epoch": checkpoint.get("epoch"),
        "z_dim": generator.z_dim,
    }
    return generator, _state["meta"]["status"]


@torch.no_grad()
def generate_images(seed: int, count: int, truncation: float) -> Image.Image:
    generator, status = load_generator()
    if generator is None:
        raise gr.Error(status)

    rng = np.random.default_rng(int(seed))
    noise = torch.from_numpy(
        rng.standard_normal((int(count), generator.z_dim)).astype(np.float32)
    )
    # truncation trades variety for typicality, the standard GAN sampling trick
    noise = noise * float(truncation)
    images = generator(noise)

    grid_columns = 4
    rows = (len(images) + grid_columns - 1) // grid_columns
    size = generator.out_size
    canvas = np.zeros((rows * size, grid_columns * size, 3), dtype=np.uint8)
    for index, image in enumerate(images):
        row, column = divmod(index, grid_columns)
        array = ((image.clamp(-1, 1) + 1) / 2 * 255).byte().cpu()
        canvas[row * size : (row + 1) * size, column * size : (column + 1) * size] = (
            array.permute(1, 2, 0).numpy()
        )
    return Image.fromarray(canvas)


def generate_video(seed_a: int, seed_b: int, frames: int, fps: int) -> str:
    generator, status = load_generator()
    if generator is None:
        raise gr.Error(status)

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SAMPLE_DIR / f"walk_{int(seed_a)}_{int(seed_b)}.mp4"
    walk = latent_walk_frames(
        generator,
        steps=int(frames),
        seed_a=int(seed_a),
        seed_b=int(seed_b),
        fps=int(fps),
    )
    written = write_video(walk, out_path, fps=int(fps))
    return str(written)


def build_ui() -> gr.Blocks:
    _, status = load_generator()
    with gr.Blocks(title="Anviksha Generator") as demo:
        gr.Markdown(
            "# Anviksha — image and video generator\n"
            "A 64×64 GAN trained from scratch. The image tab samples new frames; "
            "the video tab walks between two points in latent space.\n\n"
            f"**Status:** {status}"
        )
        with gr.Tab("Images"):
            with gr.Row():
                seed = gr.Slider(0, 9999, value=0, step=1, label="seed")
                count = gr.Slider(1, 16, value=8, step=1, label="images")
                truncation = gr.Slider(
                    0.3, 1.5, value=1.0, step=0.05,
                    label="truncation (lower = safer, higher = more varied)",
                )
            button = gr.Button("Generate", variant="primary")
            gallery = gr.Image(type="pil", label="output")
            button.click(generate_images, [seed, count, truncation], gallery)

        with gr.Tab("Video"):
            gr.Markdown(
                "Interpolates between two latent vectors and decodes every step. "
                "This is motion in latent space, not a learned temporal model — "
                "nothing here understands what should happen next."
            )
            with gr.Row():
                seed_a = gr.Slider(0, 9999, value=0, step=1, label="start seed")
                seed_b = gr.Slider(0, 9999, value=1, step=1, label="end seed")
                frames = gr.Slider(8, 96, value=48, step=4, label="frames")
                fps = gr.Slider(6, 30, value=12, step=1, label="fps")
            make_video = gr.Button("Generate video", variant="primary")
            video = gr.Video(label="output")
            make_video.click(generate_video, [seed_a, seed_b, frames, fps], video)
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
