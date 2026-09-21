"""CLIP image encoder."""

from collections.abc import Sequence
from typing import override

import torch
from torch.nn import functional as F

from anytraverse import typing as anyt
from anytraverse.interfaces import ImageEncoder
from anytraverse.models._base import HuggingFaceModel, require_transformers, to_pil_list


class CLIPImageEncoder(HuggingFaceModel, ImageEncoder):
    """
    Scene encodings from the CLIP vision tower, as used in the paper.

    Args:
        model_id (str): CLIP checkpoint. Defaults to ``"openai/clip-vit-base-patch32"``.
        device (str | torch.device | None): Device to run on; auto-detected if ``None``.
        dtype (torch.dtype | None): Weight precision; see :class:`HuggingFaceModel`.
        cache_dir (str | None): Override the Hugging Face cache directory.
        compile (bool): Wrap the model with ``torch.compile``.
    """

    def __init__(
        self,
        model_id: str = "openai/clip-vit-base-patch32",
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        cache_dir: str | None = None,
        compile: bool = False,
    ) -> None:
        super().__init__(model_id, device=device, dtype=dtype, cache_dir=cache_dir, compile=compile)
        transformers = require_transformers()
        self._processor = self._load_processor(transformers.AutoImageProcessor)
        self._model = self._load_model(transformers.CLIPVisionModelWithProjection)
        self.dim = int(self._model.config.projection_dim)

    @override
    @torch.inference_mode()
    def __call__(self, x: anyt.Image | Sequence[anyt.Image]) -> anyt.Encoding:
        inputs = self._to_device(self._processor(images=to_pil_list(x), return_tensors="pt"))
        embeds = self._model(**inputs).image_embeds
        return F.normalize(embeds.float(), p=2, dim=-1)


__all__ = ["CLIPImageEncoder"]
