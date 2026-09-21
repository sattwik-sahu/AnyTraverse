"""Benchmark attention mappings and encoders: latency and peak VRAM per frame.

Usage:
    python scripts/bench.py --models clipseg sam3 --size 640 480 --frames 20

Real weights are downloaded from the Hugging Face Hub on first use. SAM 3 is
gated: accept the license and run `hf auth login` first.
"""

from __future__ import annotations

import argparse
import time

import torch
from PIL import Image

from anytraverse import models

ATTENTION_MODELS = {
    "clipseg": lambda device, dtype: models.CLIPSegAttentionMapping(device=device, dtype=dtype),
    "sam3": lambda device, dtype: models.SAM3AttentionMapping(device=device, dtype=dtype),
    "sam3-560": lambda device, dtype: models.SAM3AttentionMapping(
        image_size=560, device=device, dtype=dtype
    ),
    "grounded-sam2": lambda device, dtype: models.GroundedSAM2AttentionMapping(
        device=device, dtype=dtype
    ),
    "owlv2-sam2": lambda device, dtype: models.OWLv2SAM2AttentionMapping(
        device=device, dtype=dtype
    ),
}

ENCODERS = {
    "clip": lambda device, dtype: models.CLIPImageEncoder(device=device, dtype=dtype),
    "siglip2": lambda device, dtype: models.SigLIP2ImageEncoder(device=device, dtype=dtype),
    "dinov2": lambda device, dtype: models.DINOv2ImageEncoder(device=device, dtype=dtype),
}

PROMPTS = ["road", "grass", "bush", "rock", "tree"]


def bench_attention(name: str, image: Image.Image, frames: int, device: str, dtype: str) -> dict:
    """Times full prompt-attention inference and records peak VRAM."""
    mapping = ATTENTION_MODELS[name](device, getattr(torch, dtype))
    prompts = PROMPTS
    mapping(image, prompts[:1])  # warmup (and CUDA-graph-free lazy init)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    latencies = []
    for _ in range(frames):
        start = time.perf_counter()
        maps = mapping(image, prompts)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)
        assert len(maps) == len(prompts)
    latencies.sort()
    vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    return {
        "name": name,
        "mean_ms": sum(latencies) / len(latencies),
        "p95_ms": latencies[int(0.95 * (len(latencies) - 1))],
        "peak_vram_gb": vram,
    }


def bench_encoder(name: str, image: Image.Image, frames: int, device: str, dtype: str) -> dict:
    """Times scene-encoder inference and records peak VRAM."""
    encoder = ENCODERS[name](device, getattr(torch, dtype))
    encoder(image)  # warmup
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    latencies = []
    for _ in range(frames):
        start = time.perf_counter()
        encoding = encoder(image)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)
        assert encoding.shape == (1, encoder.dim)
    latencies.sort()
    vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    return {
        "name": f"encoder:{name}",
        "mean_ms": sum(latencies) / len(latencies),
        "p95_ms": latencies[int(0.95 * (len(latencies) - 1))],
        "peak_vram_gb": vram,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["clipseg"],
        choices=[*ATTENTION_MODELS, *[f"encoder:{e}" for e in ENCODERS]],
    )
    parser.add_argument("--size", nargs=2, type=int, default=[640, 480], metavar=("W", "H"))
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="float16" if torch.cuda.is_available() else "float32")
    args = parser.parse_args()

    from anytraverse.device import resolve_device

    device = resolve_device(args.device)
    image = Image.new("RGB", tuple(args.size), (110, 130, 90))

    rows = []
    for name in args.models:
        if name.startswith("encoder:"):
            rows.append(
                bench_encoder(name.split(":", 1)[1], image, args.frames, device, args.dtype)
            )
        else:
            rows.append(bench_attention(name, image, args.frames, device, args.dtype))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n# {args.size[0]}x{args.size[1]}, {args.frames} frames, {device}, {args.dtype}")
    print(f"| {'model':<16} | {'mean ms':>8} | {'p95 ms':>8} | {'peak VRAM GB':>12} |")
    print(f"| {'-' * 16} | {'-' * 8} | {'-' * 8} | {'-' * 12} |")
    for row in rows:
        print(
            f"| {row['name']:<16} | {row['mean_ms']:>8.1f} | {row['p95_ms']:>8.1f} "
            f"| {row['peak_vram_gb']:>12.2f} |"
        )


if __name__ == "__main__":
    main()
