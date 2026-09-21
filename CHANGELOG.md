# Changelog

## 2.0.0

### Breaking

- Flat package layout: `from anytraverse import AnyTraverse, CLIPSegAttentionMapping`.
  The `anytraverse.utils.*` and `anytraverse.helpers.*` paths are gone.
- Spelling fixes: `TraversalState.UNKNOWN_OBJECT` (was `UNKOWN_OBJ`),
  `init_traversability_preferences` (was `init_traversabilty_preferences`),
  `traversability_preferences` (was `traversabilty_preferences`).
- Removed the Nomic encoder; use `siglip2` or `dinov2`.
- `requires-python = ">=3.12"`; model weights use the standard Hugging Face cache
  instead of `data/weights/`.

### Added

- SAM 3 (`SAM3AttentionMapping`, `facebook/sam3`) with semantic/instance modes and an
  `image_size` knob for low-memory devices.
- Detector + SAM 2 mappings: `GroundedSAM2AttentionMapping`
  (`IDEA-Research/grounding-dino-tiny`) and `OWLv2SAM2AttentionMapping`
  (`google/owlv2-base-patch16-ensemble`), both with `facebook/sam2.1-hiera-tiny`.
- Scene encoders: `SigLIP2ImageEncoder` and `DINOv2ImageEncoder`.
- Presets: `build_pipeline`, `build_pipeline_from_paper`, `build_pipeline_sam3`,
  `build_pipeline_grounded_sam2`, all with `device`/`dtype` arguments.
- `scripts/bench.py` reporting latency and peak VRAM per model.
- Full test suite at 100% coverage, run in CI on Python 3.12/3.13 with a
  `ruff`-only lint gate and a pre-commit config.

### Performance

- `AnyTraverse.step` runs under `torch.inference_mode`.
- Prompt text embeddings are cached; the image is encoded once per frame no matter
  how many prompts are used.
- `EncodingHistory` keeps a stacked encoding tensor, so a lookup is one similarity call.
- Every model accepts `dtype` (half precision on CUDA) and opt-in `compile`.
- Fixed `CLIPImageEncoder` ignoring the device; removed the per-call `Resize` layer
  and the `[image] * len(prompts)` batch replication.

### Fixed

- `parse_trav_pref_syntax` no longer crashes on trailing semicolons and reports the
  offending segment.
- `Threshold` and the preferences setter raise `ValueError` instead of asserting.
- `ProbabilisticTraversabilityPooler` handles all-positive/all-negative prompt sets.
- `AnyTraverseState.to(device)` moves maps off GPU, e.g. before ROS publishing.
