"""SigLIP 2 image encoder."""

from collections.abc import Sequence
from typing import override

import torch
from torch.nn import functional as F

from anytraverse import typing as anyt
from anytraverse.interfaces import ImageEncoder
from anytraverse.models._base import HuggingFaceModel, require_transformers, to_pil_list


class SigLIP2ImageEncoder(HuggingFaceModel, ImageEncoder):
    """
    Scene encodings from the SigLIP 2 vision tower.

    SigLIP 2 gives noticeably better image retrieval than CLIP ViT-B/32 at a
    similar size, which makes scene matching in the history more reliable.

    Args:
        model_id (str): SigLIP 2 checkpoint. Defaults to
            ``"google/siglip2-base-patch16-naflex"``. Note that the fixed-size
            ``siglip2-base-patch16-{224,256,384,512}`` checkpoints are the older
            SigLIP architecture despite their names; the ``naflex`` variants are
            genuine SigLIP 2 and accept any input resolution.
        device (str | torch.device | None): Device to run on; auto-detected if ``None``.
        dtype (torch.dtype | None): Weight precision; see :class:`HuggingFaceModel`.
        cache_dir (str | None): Override the Hugging Face cache directory.
        compile (bool): Wrap the model with ``torch.compile``.
    """

    def __init__(
        self,
        model_id: str = "google/siglip2-base-patch16-naflex",
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        cache_dir: str | None = None,
        compile: bool = False,
    ) -> None:
        super().__init__(model_id, device=device, dtype=dtype, cache_dir=cache_dir, compile=compile)
        transformers = require_transformers()
        self._processor = self._load_processor(transformers.AutoImageProcessor)
        # The full model class loads the checkpoint cleanly; the vision-only
        # class does not match its embedding layout.
        self._model = self._load_model(transformers.Siglip2Model)
        self.dim = int(self._model.vision_model.config.hidden_size)

    @override
    @torch.inference_mode()
    def __call__(self, x: anyt.Image | Sequence[anyt.Image]) -> anyt.Encoding:
        inputs = self._to_device(self._processor(images=to_pil_list(x), return_tensors="pt"))
        # NaFlex checkpoints additionally need the per-image spatial shapes.
        extra = {
            key: inputs[key] for key in ("pixel_attention_mask", "spatial_shapes") if key in inputs
        }
        pooled = self._model.get_image_features(pixel_values=inputs["pixel_values"], **extra)
        return F.normalize(pooled.pooler_output.float(), p=2, dim=-1)


__all__ = ["SigLIP2ImageEncoder"]
