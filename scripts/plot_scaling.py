"""Plots step time vs. number of prompts from ``assets/prompt_scaling.csv``::

python scripts/plot_scaling.py --csv assets/prompt_scaling.csv --out assets/prompt_scaling.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

MODEL_STYLE = {
    "CLIPSeg": {"color": "#0173b2", "marker": "o"},
    "SAM3-1008": {"color": "#de8f05", "marker": "s"},
    "SAM3-560": {"color": "#cc78bc", "marker": "D"},
    "Grounded-SAM2": {"color": "#029e73", "marker": "^"},
    "OWLv2-SAM2": {"color": "#d55e00", "marker": "v"},
}


def load(csv_path: Path) -> dict[str, dict[str, list[float]]]:
    """Returns ``{model: {"x": [...], "mean": [...], "std": [...]}}`` sorted by prompt count."""
    data: dict[str, list[tuple]] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            data.setdefault(row["model"], []).append(
                (int(row["prompts"]), float(row["mean_ms"]), float(row["std_ms"]))
            )
    out = {}
    for model, points in data.items():
        points.sort()
        out[model] = {
            "x": [p[0] for p in points],
            "mean": [p[1] for p in points],
            "std": [p[2] for p in points],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="assets/prompt_scaling.csv")
    parser.add_argument("--out", default="assets/prompt_scaling.png")
    args = parser.parse_args()

    data = load(Path(args.csv))

    plt.style.use("seaborn-v0_8-colorblind")
    fig, ax = plt.subplots(figsize=(10, 6))
    for model, series in data.items():
        style = MODEL_STYLE.get(model, {"marker": "o"})
        ax.errorbar(
            series["x"],
            series["mean"],
            yerr=series["std"],
            label=model,
            capsize=3,
            elinewidth=1.2,
            linewidth=2.2,
            markersize=6,
            **style,
        )
    ax.set_xlabel("Number of prompts", fontsize=12)
    ax.set_ylabel("Step time per frame (ms)", fontsize=12)
    ax.set_title(
        "AnyTraverse step time vs. number of prompts (RTX A4500, 640x480, fp16)", fontsize=13
    )
    ax.set_xticks(sorted({x for s in data.values() for x in s["x"]}))
    ax.grid(True, which="major", linestyle="--", alpha=0.5)
    ax.legend(title="Attention model", fontsize=11, title_fontsize=11, loc="upper left")
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
