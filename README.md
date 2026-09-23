# AnyTraverse

An offroad traversability framework with a VLM and a human operator in the loop.

[![arXiv](https://img.shields.io/badge/arXiv-2506.16826v1-b31b1b.svg?logo=arxiv&style=flat)](https://arxiv.org/abs/2506.16826v1)
[![pypi](https://img.shields.io/pypi/v/anytraverse?style=flat&logo=python)](https://pypi.org/project/anytraverse/)
[![CI](https://github.com/sattwik-sahu/AnyTraverse/actions/workflows/ci.yml/badge.svg)](https://github.com/sattwik-sahu/AnyTraverse/actions/workflows/ci.yml)
![GitHub Repo stars](https://img.shields.io/github/stars/sattwik-sahu/anytraverse)

## News :newspaper:

- **Jul, 2025:** AnyTraverse presented at the *19th International Symposium on Experimental Robotics*, Santa Fe, New Mexico. Set to be published in proceedings (Springer Nature, "Experimental Robotics").

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
   pip install "anytraverse[viz]"         # plotting helpers for the examples below
   ```
   SAM 3 weights are gated: accept the license for [`facebook/sam3`](https://huggingface.co/facebook/sam3) and run `hf auth login` once.

---

## Usage

### Quickstart :running_man:

The pipeline from the [paper](https://arxiv.org/abs/2506.16826v1): CLIPSeg attention maps, CLIP scene encodings, and the paper's poolers.

```python
from anytraverse import build_pipeline_from_paper
from PIL import Image as PILImage
import requests


def main():
    url = (
        "https://source.roboflow.com/oWTBJ1yeWRbHDXbzJBrOsPVaoH92/0C8goYvWpiqF26dNKxby/original.jpg"
    )
    image = PILImage.open(requests.get(url, stream=True).raw)

    pipeline = build_pipeline_from_paper(
        init_traversability_preferences={"road": 1, "bush": -0.8, "rock": 0.45},
        ref_scene_similarity_threshold=0.8,
        roi_uncertainty_threshold=0.3,
        roi_x_bounds=(0.333, 0.667),
        roi_y_bounds=(0.6, 0.95),
    )
    state = pipeline.step(image)
    print(state.traversal_state, state.roi_traversability, state.roi_uncertainty)


if __name__ == "__main__":
    main()
```

`state` holds everything computed for the frame: per-prompt `attention_maps`, the `traversability_map` and `uncertainty_map`, their ROI crops, `ref_scene_similarity`, `roi_traversability`, `roi_uncertainty`, and the `traversal_state` decision (`OK`, `UNKNOWN_SCENE` or `UNKNOWN_OBJECT`).

See the [extended example](assets/) for plotting attention, traversability and uncertainty maps:

_Attention maps_
![](./assets/attention_maps.png)

_Traversability and uncertainty maps_
![](./assets/trav_unc_maps.png)

### Other VLMs and encoders

Swap the attention model, the scene encoder, or both. All weights below are openly available on the Hugging Face Hub and fit in 10 GB of VRAM in half precision. See [benchmarks.md](benchmarks.md) for measured latency and memory on an RTX A4500 — e.g. the paper pipeline runs in ~8 ms in ~0.6 GB, and the heaviest SAM 3 combination peaks around 3 GB. There is also a [prompt-scaling plot](benchmarks.md#step-time-vs-number-of-prompts) showing step time for 1–10 prompts.

```python
from anytraverse import build_pipeline_grounded_sam2, build_pipeline_sam3

# SAM 3 attention maps with a SigLIP2 scene encoder; image_size=560 halves memory use.
sam3 = build_pipeline_sam3(
    init_traversability_preferences={"road": 1, "bush": -0.8},
    ref_scene_similarity_threshold=0.8,
    roi_uncertainty_threshold=0.3,
    encoder="siglip2",  # or "dinov2" or "clip"
    image_size=560,
)

# Grounding DINO detections refined by SAM 2 (detector="owlv2" for the OWLv2 variant).
grounded = build_pipeline_grounded_sam2(
    init_traversability_preferences={"road": 1, "bush": -0.8},
    ref_scene_similarity_threshold=0.8,
    roi_uncertainty_threshold=0.3,
)
```

For full control, assemble `AnyTraverse` directly from any `PromptAttentionMapping`, `ImageEncoder` and pooler classes — see `anytraverse.presets.build_pipeline` and the interface docs in `anytraverse.interfaces`.

### Operator in the loop

When `state.traversal_state` is `UNKNOWN_SCENE` or `UNKNOWN_OBJECT`, ask the operator and feed the answer back with the `"prompt: weight; ..."` syntax. An empty call just registers the current scene.

```python
pipeline.human_call("mud: -0.5; road: 0.9")
pipeline.register_scene()  # no preference change
```

Revisited scenes are recalled from the history automatically, so the operator is only asked about genuinely new situations.

### Deployment notes

- Every model takes `device`, `dtype` and `compile` arguments. On CUDA, half precision (`dtype=torch.float16`) roughly halves memory and latency; `AnyTraverse.step` runs under `torch.inference_mode`.
- Prompt text embeddings are cached per prompt set and the image is encoded once per frame no matter how many prompts you use.
- Call `state.to("cpu")` before publishing maps over ROS; weights are cached in the standard Hugging Face cache (`HF_HOME`).
- Measure your board with `python scripts/bench.py --models clipseg sam3 sam3-560 grounded-sam2 owlv2-sam2 --size 640 480`.

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
