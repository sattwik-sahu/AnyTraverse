"""Step time vs. number of prompts for every attention mapping.

Runs each model with 1..N prompts, K timed frames each (after warmup), and
writes the raw numbers to ``assets/prompt_scaling.csv``::

    python scripts/bench_scaling.py --max-prompts 10 --frames 10
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from PIL import Image

from anytraverse import models
from anytraverse.device import resolve_device

PROMPTS = ["road", "grass", "bush", "rock", "tree", "mud", "sand", "gravel", "puddle", "fence"]

ATTENTION_MODELS = {
    "CLIPSeg": lambda device, dtype: models.CLIPSegAttentionMapping(device=device, dtype=dtype),
    "SAM3-1008": lambda device, dtype: models.SAM3AttentionMapping(device=device, dtype=dtype),
    "SAM3-560": lambda device, dtype: models.SAM3AttentionMapping(
        image_size=560, device=device, dtype=dtype
    ),
    "Grounded-SAM2": lambda device, dtype: models.GroundedSAM2AttentionMapping(
        device=device, dtype=dtype
    ),
    "OWLv2-SAM2": lambda device, dtype: models.OWLv2SAM2AttentionMapping(
        device=device, dtype=dtype
    ),
}


def time_frames(
    mapping, image: Image.Image, prompts: list[str], frames: int, warmup: int = 5
) -> list[float]:
    """Warms up on the exact prompt count, then returns per-frame latencies in milliseconds."""
    for _ in range(warmup):
        mapping(image, prompts)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latencies = []
    for _ in range(frames):
        start = time.perf_counter()
        maps = mapping(image, prompts)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        assert len(maps) == len(prompts)
        latencies.append((time.perf_counter() - start) * 1000.0)
    return latencies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-prompts", type=int, default=10)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--size", nargs=2, type=int, default=[640, 480], metavar=("W", "H"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="float16" if torch.cuda.is_available() else "float32")
    parser.add_argument("--out", default="assets/prompt_scaling.csv")
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = getattr(torch, args.dtype)
    image = Image.new("RGB", tuple(args.size), (110, 130, 90))
    prompts = PROMPTS[: args.max_prompts]

    rows: list[dict] = []
    for name, factory in ATTENTION_MODELS.items():
        print(f"{name} ...", flush=True)
        mapping = factory(device, dtype)
        for n in range(1, len(prompts) + 1):
            latencies = time_frames(mapping, image, prompts[:n], args.frames)
            mean = sum(latencies) / len(latencies)
            std = (sum((x - mean) ** 2 for x in latencies) / len(latencies)) ** 0.5
            rows.append({"model": name, "prompts": n, "mean_ms": mean, "std_ms": std})
            print(f"  {n:2d} prompts: {mean:7.1f} ± {std:5.1f} ms", flush=True)
        del mapping
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "prompts", "mean_ms", "std_ms"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
