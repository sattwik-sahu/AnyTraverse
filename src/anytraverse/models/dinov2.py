"""DINOv2 image encoder."""

from collections.abc import Sequence
from typing import override

import torch
from torch.nn import functional as F

from anytraverse import typing as anyt
from anytraverse.interfaces import ImageEncoder
from anytraverse.models._base import HuggingFaceModel, require_transformers, to_pil_list


class DINOv2ImageEncoder(HuggingFaceModel, ImageEncoder):
    """
    Scene encodings from the DINOv2 ``[CLS]`` token.

    DINOv2 is purely visual (no text alignment), which makes it a strong choice
    for recognising a place the robot has seen before.

    Args:
        model_id (str): DINOv2 checkpoint. Defaults to ``"facebook/dinov2-small"``.
        device (str | torch.device | None): Device to run on; auto-detected if ``None``.
        dtype (torch.dtype | None): Weight precision; see :class:`HuggingFaceModel`.
        cache_dir (str | None): Override the Hugging Face cache directory.
        compile (bool): Wrap the model with ``torch.compile``.
    """

    def __init__(
        self,
        model_id: str = "facebook/dinov2-small",
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        cache_dir: str | None = None,
        compile: bool = False,
    ) -> None:
        super().__init__(model_id, device=device, dtype=dtype, cache_dir=cache_dir, compile=compile)
        transformers = require_transformers()
        self._processor = self._load_processor(transformers.AutoImageProcessor)
        self._model = self._load_model(transformers.Dinov2Model)
        self.dim = int(self._model.config.hidden_size)

    @override
    @torch.inference_mode()
    def __call__(self, x: anyt.Image | Sequence[anyt.Image]) -> anyt.Encoding:
        inputs = self._to_device(self._processor(images=to_pil_list(x), return_tensors="pt"))
        cls_token = self._model(**inputs).last_hidden_state[:, 0]
        return F.normalize(cls_token.float(), p=2, dim=-1)


__all__ = ["DINOv2ImageEncoder"]
