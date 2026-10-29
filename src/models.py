"""Generator and discriminator for 64x64 image synthesis, plus a latent-walk
video sampler.

Both networks are deliberately small. The target deployment is a CPU-only
Hugging Face Space (2 vCPU, 16 GB), so the whole generator fits comfortably in
memory and produces a 64x64 image in well under a second.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class Generator(nn.Module):
    """Maps a latent vector to a 64x64 RGB image in [-1, 1]."""

    def __init__(self, z_dim: int = 128, base: int = 32, out_size: int = 64):
        super().__init__()
        if out_size not in (32, 64, 128):
            raise ValueError("out_size must be 32, 64 or 128")
        upsamples = {32: 3, 64: 4, 128: 5}[out_size]

        layers: list[nn.Module] = [
            nn.ConvTranspose2d(z_dim, base * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(True),
        ]
        width = base * 8
        for _ in range(upsamples - 1):
            next_width = max(width // 2, base)
            layers += [
                nn.ConvTranspose2d(width, next_width, 4, 2, 1, bias=False),
                nn.BatchNorm2d(next_width),
                nn.ReLU(True),
            ]
            width = next_width
        layers += [nn.ConvTranspose2d(width, 3, 4, 2, 1), nn.Tanh()]

        self.net = nn.Sequential(*layers)
        self.z_dim = z_dim
        self.out_size = out_size

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.view(len(z), self.z_dim, 1, 1))


class Discriminator(nn.Module):
    """Scores a 64x64 image as real or generated."""

    def __init__(self, base: int = 32, in_size: int = 64):
        super().__init__()
        downsamples = {32: 3, 64: 4, 128: 5}[in_size]

        layers: list[nn.Module] = [
            nn.Conv2d(3, base, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
        ]
        width = base
        for _ in range(downsamples - 1):
            next_width = min(width * 2, base * 8)
            layers += [
                nn.Conv2d(width, next_width, 4, 2, 1, bias=False),
                nn.BatchNorm2d(next_width),
                nn.LeakyReLU(0.2, True),
            ]
            width = next_width
        self.features = nn.Sequential(*layers)
        reduced = in_size // (2**downsamples)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(width * reduced * reduced, 1)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(images))


def save_image_grid(images: torch.Tensor, path: Path, columns: int = 4) -> None:
    """Write a tensor of images in [-1, 1] to a PNG grid."""
    from PIL import Image

    images = ((images.clamp(-1, 1) + 1) / 2 * 255).byte().cpu()
    count, _, height, width = images.shape
    rows = (count + columns - 1) // columns
    canvas = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index in range(count):
        row, column = divmod(index, columns)
        canvas[
            row * height : (row + 1) * height, column * width : (column + 1) * width
        ] = images[index].permute(1, 2, 0).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(path)


@torch.no_grad()
def latent_walk_frames(
    generator: Generator,
    steps: int = 48,
    seed_a: int = 0,
    seed_b: int = 1,
    fps: int = 12,
    device: str = "cpu",
) -> list[np.ndarray]:
    """Interpolate between two latent vectors and decode every step.

    This produces motion in latent space. It is not a learned temporal model:
    there is no notion of physics or of what should happen next, only a smooth
    path between two points the generator already knows how to draw.
    """
    generator.eval().to(device)
    rng = np.random.default_rng(seed_a)
    start = torch.from_numpy(
        rng.standard_normal((1, generator.z_dim)).astype(np.float32)
    ).to(device)
    rng = np.random.default_rng(seed_b)
    end = torch.from_numpy(
        rng.standard_normal((1, generator.z_dim)).astype(np.float32)
    ).to(device)

    frames: list[np.ndarray] = []
    for step in range(steps):
        weight = step / max(steps - 1, 1)
        # spherical interpolation keeps the path on the hypersphere, which
        # avoids the washed-out frames a plain linear blend produces
        blend = _slerp(start, end, weight)
        image = generator(blend)[0]
        array = ((image.clamp(-1, 1) + 1) / 2 * 255).byte().cpu()
        frames.append(array.permute(1, 2, 0).numpy())
    return frames


def _slerp(a: torch.Tensor, b: torch.Tensor, weight: float) -> torch.Tensor:
    a_norm = a / a.norm(dim=1, keepdim=True)
    b_norm = b / b.norm(dim=1, keepdim=True)
    cosine = (a_norm * b_norm).sum(dim=1, keepdim=True).clamp(-1, 1)
    omega = torch.acos(cosine)
    sine = torch.sin(omega)
    safe_sine = torch.where(sine.abs() < 1e-6, torch.ones_like(sine), sine)
    first = torch.sin((1 - weight) * omega) / safe_sine
    second = torch.sin(weight * omega) / safe_sine
    return first * a + second * b


def write_video(frames: list[np.ndarray], path: Path, fps: int = 12) -> Path:
    """Write frames to an MP4, falling back to GIF when no encoder is present."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio

        imageio.mimsave(path, frames, fps=fps, macro_block_size=1)
        return path
    except Exception:
        from PIL import Image

        gif_path = path.with_suffix(".gif")
        pil_frames = [Image.fromarray(frame) for frame in frames]
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000 / fps),
            loop=0,
        )
        return gif_path
