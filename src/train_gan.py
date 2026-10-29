"""Train the 64x64 GAN.

Designed to run on a CPU-only Hugging Face Space (2 vCPU, 16 GB RAM) or on any
laptop. A run of 40 epochs over ~1,200 images takes roughly an hour on 2 cores.

    python src/train_gan.py --data data/images --epochs 40 --out weights

Checkpoints store the generator, discriminator, optimiser state and the epoch
counter, so a run that is stopped (or a Space that restarts) resumes cleanly.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import Discriminator, Generator, save_image_grid  # noqa: E402

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/images", type=Path)
    parser.add_argument("--out", default="weights", type=Path)
    parser.add_argument("--size", default=64, type=int)
    parser.add_argument("--z-dim", default=128, type=int)
    parser.add_argument("--base", default=32, type=int)
    parser.add_argument("--batch", default=16, type=int)
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--threads", default=2, type=int)
    parser.add_argument("--time-budget", default=0.0, type=float)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_images(root: Path, size: int) -> np.ndarray:
    """Load every image under `root` into a uint8 array of shape (N, H, W, 3)."""
    paths = [p for p in sorted(root.rglob("*")) if p.suffix.lower() in IMAGE_SUFFIXES]
    if not paths:
        raise SystemExit(f"no images found under {root}")
    array = np.stack(
        [
            np.asarray(
                Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR),
                dtype=np.uint8,
            )
            for path in paths
        ]
    )
    print(f"loaded {len(array)} images from {root}")
    return array


def to_tensor(batch: np.ndarray) -> torch.Tensor:
    """uint8 NHWC -> float32 NCHW scaled to [-1, 1]."""
    return torch.from_numpy(batch.astype(np.float32).transpose(0, 3, 1, 2) / 127.5 - 1.0)


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    images = load_images(args.data, args.size)
    generator = Generator(z_dim=args.z_dim, base=args.base, out_size=args.size)
    discriminator = Discriminator(base=args.base, in_size=args.size)
    print(
        f"generator {sum(p.numel() for p in generator.parameters()):,} params, "
        f"discriminator {sum(p.numel() for p in discriminator.parameters()):,} params"
    )

    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out / "gan.pt"
    start_epoch = 0
    history: list[dict] = []
    fixed_noise = torch.randn(16, args.z_dim, generator=torch.Generator().manual_seed(7))

    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        opt_g.load_state_dict(checkpoint["opt_g"])
        opt_d.load_state_dict(checkpoint["opt_d"])
        start_epoch = checkpoint["epoch"]
        history = checkpoint["history"]
        print(f"resumed at epoch {start_epoch}")

    started = time.time()
    for epoch in range(start_epoch, args.epochs):
        generator.train()
        discriminator.train()
        order = np.random.default_rng(epoch).permutation(len(images))
        loss_d_total = 0.0
        loss_g_total = 0.0
        steps = 0

        for start in range(0, len(order), args.batch):
            index = order[start : start + args.batch]
            if len(index) < 2:
                continue
            real = to_tensor(images[index])
            count = len(real)
            ones = torch.ones(count, 1)
            zeros = torch.zeros(count, 1)

            # discriminator step
            opt_d.zero_grad()
            noise = torch.randn(count, args.z_dim)
            fake = generator(noise)
            loss_d = 0.5 * (
                criterion(discriminator(real), ones)
                + criterion(discriminator(fake.detach()), zeros)
            )
            loss_d.backward()
            opt_d.step()

            # generator step
            opt_g.zero_grad()
            loss_g = criterion(discriminator(fake), ones)
            loss_g.backward()
            opt_g.step()

            loss_d_total += loss_d.item()
            loss_g_total += loss_g.item()
            steps += 1

        mean_d = loss_d_total / max(steps, 1)
        mean_g = loss_g_total / max(steps, 1)
        history.append({"epoch": epoch + 1, "loss_d": mean_d, "loss_g": mean_g})
        print(
            f"epoch {epoch + 1:3d}/{args.epochs}  loss_d {mean_d:.4f}  "
            f"loss_g {mean_g:.4f}  ({time.time() - started:.0f}s)",
            flush=True,
        )

        generator.eval()
        with torch.no_grad():
            samples = generator(fixed_noise)
        save_image_grid(samples, args.out / f"epoch_{epoch + 1:03d}.png")

        torch.save(
            {
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict(),
                "epoch": epoch + 1,
                "history": history,
                "z_dim": args.z_dim,
                "base": args.base,
                "size": args.size,
            },
            checkpoint_path,
        )
        (args.out / "metrics.json").write_text(
            json.dumps(
                {"epochs_completed": len(history), "history": history}, indent=2
            )
        )

        if args.time_budget and time.time() - started > args.time_budget:
            print("time budget reached; rerun with --resume to continue")
            break

    print(f"checkpoint written to {checkpoint_path}")


if __name__ == "__main__":
    main()
