"""Shared plumbing for the Hugging Face model wrappers."""

import importlib
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image as PILImage
from torch.nn import functional as F

from anytraverse import typing as anyt
from anytraverse.device import get_default_dtype, resolve_device


def require_transformers() -> Any:
    """
    Imports ``transformers`` lazily with a helpful error message.

    Returns:
        The ``transformers`` module.

    Raises:
        ImportError: If the ``hf`` extra is not installed.
    """
    try:
        return importlib.import_module("transformers")
    except ImportError as exc:  # pragma: no cover - needs transformers uninstalled
        raise ImportError(
            "The Hugging Face model wrappers need the 'hf' extra: pip install 'anytraverse[hf]'"
        ) from exc


def to_pil(x: anyt.Image) -> PILImage.Image:
    """
    Converts an image to an RGB PIL image.

    Args:
        x (Image): A PIL image or a ``uint8`` array of shape ``(H, W, 3)``.

    Returns:
        PIL.Image.Image: The image in RGB mode.
    """
    if isinstance(x, np.ndarray):
        x = PILImage.fromarray(x)
    return x.convert("RGB") if x.mode != "RGB" else x


def to_pil_list(x: anyt.Image | Sequence[anyt.Image]) -> list[PILImage.Image]:
    """Converts a single image or a sequence of images to a list of RGB PIL images."""
    if isinstance(x, PILImage.Image | np.ndarray):
        return [to_pil(x)]
    return [to_pil(img) for img in x]


def resize_maps(maps: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """
    Bilinearly resizes a stack of maps.

    Args:
        maps (torch.Tensor): Tensor of shape ``(N, h, w)``.
        size (tuple[int, int]): Target ``(H, W)``.

    Returns:
        torch.Tensor: Tensor of shape ``(N, H, W)``.
    """
    if tuple(maps.shape[-2:]) == tuple(size):
        return maps
    return F.interpolate(
        maps.unsqueeze(1).float(), size=size, mode="bilinear", align_corners=False
    ).squeeze(1)


class HuggingFaceModel:
    """
    Base class holding device, dtype and loading options for a Hugging Face model.

    Args:
        model_id (str): Model repository on the Hugging Face Hub.
        device (str | torch.device | None): Device to run on; auto-detected if ``None``.
        dtype (torch.dtype | None): Weight precision; ``float16`` on CUDA and
            ``float32`` elsewhere if ``None``.
        cache_dir (str | None): Override the Hugging Face cache directory.
        compile (bool): Wrap the model with ``torch.compile``. Off by default
            because it is slow to warm up and unreliable on some embedded boards.
    """

    def __init__(
        self,
        model_id: str,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        cache_dir: str | None = None,
        compile: bool = False,
    ) -> None:
        self.model_id = model_id
        self.device = resolve_device(device)
        self.dtype = dtype if dtype is not None else get_default_dtype(self.device)
        self.cache_dir = cache_dir
        self.compile = compile

    def _load_model(self, model_cls: Any, **kwargs: Any) -> Any:
        """
        Loads a model with the wrapper's device, dtype and cache settings.

        Args:
            model_cls: A ``transformers`` model class with ``from_pretrained``.
            **kwargs: Extra arguments forwarded to ``from_pretrained``.

        Returns:
            The model in eval mode on ``self.device``.
        """
        model = model_cls.from_pretrained(
            self.model_id, dtype=self.dtype, cache_dir=self.cache_dir, **kwargs
        )
        model = model.to(self.device).eval()
        if self.compile and hasattr(torch, "compile"):
            model = torch.compile(model, mode="reduce-overhead")
        return model

    def _load_processor(self, processor_cls: Any, **kwargs: Any) -> Any:
        """Loads a processor with the wrapper's cache settings."""
        return processor_cls.from_pretrained(self.model_id, cache_dir=self.cache_dir, **kwargs)

    def _to_device(self, inputs: Any) -> Any:
        """Moves a ``BatchEncoding``/``BatchFeature`` to the device, casting floats to ``dtype``."""
        moved = inputs.to(self.device)
        for key, value in list(moved.items()):
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                moved[key] = value.to(self.dtype)
        return moved


def as_prompt_list(prompts: anyt.Prompt | Sequence[anyt.Prompt]) -> list[anyt.Prompt]:
    """Normalizes a single prompt or a sequence of prompts to a list."""
    if isinstance(prompts, str):
        return [prompts]
    prompts = list(prompts)
    if not prompts:
        raise ValueError("At least one prompt is required")
    return prompts


__all__ = [
    "HuggingFaceModel",
    "as_prompt_list",
    "require_transformers",
    "resize_maps",
    "to_pil",
    "to_pil_list",
]
