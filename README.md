# AnyTraverse

An offroad traversability framework with a VLM and a human operator in the loop.

[![arXiv](https://img.shields.io/badge/arXiv-2506.16826v1-b31b1b.svg?logo=arxiv&style=flat)](https://arxiv.org/abs/2506.16826v1)
[![pypi](https://img.shields.io/pypi/v/anytraverse?style=flat&logo=python)](https://pypi.org/project/anytraverse/)
[![CI](https://github.com/sattwik-sahu/AnyTraverse/actions/workflows/ci.yml/badge.svg)](https://github.com/sattwik-sahu/AnyTraverse/actions/workflows/ci.yml)
![GitHub Repo stars](https://img.shields.io/github/stars/sattwik-sahu/anytraverse)

## News :newspaper:

- **Sep 2026**: Released `v2` with improved API, more VLMs, and benchmarking data. 
- **Jul 2025:** AnyTraverse presented at the *19th International Symposium on Experimental Robotics*, Santa Fe, New Mexico. Set to be published in proceedings (Springer Nature, "Experimental Robotics").

## Installation

Requires Python 3.12 or newer.

1. **Install PyTorch first**, so you get the right build for your platform (CUDA, MPS, CPU, or a Jetson wheel).
   ```bash
   uv pip install torch       # uv users
   pip install torch          # pip users
   ```
   > :warning: _PyTorch does not provide wheels for the NVIDIA Jetson platform._ On Jetson, install the matching `torch` wheel first: JetPack 6 via `https://pypi.jetson-ai-lab.io/jp6/cu126`, JetPack 7 via `https://pypi.jetson-ai-lab.io/sbsa/cu132`. The `hf` extra below then pulls a matching `torchvision`.
2. **Install AnyTraverse with the model backends you need.**
   ```bash
   uv pip install "anytraverse[hf]"       # uv users (Hugging Face VLMs)
   pip install "anytraverse[hf]"          # pip users (Hugging Face VLMs)
   pip install anytraverse                # core only, for custom models
   pip install "anytraverse[viz]"         # plotting helpers for the quickstart tutorial
   ```
   SAM 3 weights are gated: accept the license for [`facebook/sam3`](https://huggingface.co/facebook/sam3) and run `hf auth login` once.

---

## Docs

- **[Quickstart](docs/quickstart.md)** :running_man: — build the paper pipeline, run one step, and plot attention, traversability and uncertainty maps. Also covers swapping VLMs/encoders (`build_pipeline_sam3`, `build_pipeline_grounded_sam2`), the operator-in-the-loop calls, and deployment notes (dtypes, caching, `state.to("cpu")` for ROS).
- **[Benchmarks](docs/benchmarks.md)** :bar_chart: — measured latency and VRAM per model on an RTX A4500, end-to-end pipeline estimates (the paper pipeline runs in ~8 ms in ~0.6 GB), and a step-time vs. prompt-count plot for 1–10 prompts.

### Launch AnyTraverse with ROS 2 :robot:

AnyTraverse integrates with the Nav2 stack in ROS 2. See [`sattwik-sahu/anytraverse_ros`](https://github.com/sattwik-sahu/anytraverse-ros) to learn more.

---

## Breaking changes in 2.0

- Flat package layout: `from anytraverse import AnyTraverse, CLIPSegAttentionMapping` instead of `anytraverse.utils.*`.
- Spelling fixes: `TraversalState.UNKNOWN_OBJECT` (was `UNKOWN_OBJ`), `init_traversability_preferences` (was `init_traversabilty_preferences`).
- The Nomic encoder was removed; use `siglip2` (default in the new presets) or `dinov2`.
- `requires-python = ">=3.12"`, `torch >= 2.4`; model weights now use the standard Hugging Face cache.

---

## Contributing :man_technologist:

We'd love to see your models and improvements. Please open a pull request (branch name: `dev/feat/<your-feature-name>`) and an issue to discuss a new feature. Run `ruff check .`, `ruff format --check .` and `pytest` before pushing.

---

Made with :heart: in IISER Bhopal.
