from abc import ABC, abstractmethod
from typing import Any, Sequence
from anytraverse import typing as anyt
from PIL import Image as PILImage
import torch
from anytraverse.helpers.device import DEVICE


class BaseEncoder[TInput](ABC):
    """
    The base encoder module.

    Encodes any input to a numeric representation, as `torch.Tensor`.
    """

    DIM: int

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def __call__(
        self, x: TInput | Sequence[TInput], *args: Any, **kwargs: Any
    ) -> anyt.Encoding:
        pass


class ImageEncoder[TImage: anyt.Image](BaseEncoder[TImage], ABC):
    """
    Base class for an image encoder module.
    """

    def __init__(self) -> None:
        super().__init__()


class HuggingFaceImageEncoder[TImage: anyt.Image](ImageEncoder[TImage], ABC):
    """
    Abstract base class for a wrapper for an image encoder from HuggingFace
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: str | None = None,
        use_fast: bool = True,
        device: torch.device = DEVICE,
    ) -> None:
        super().__init__()
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._use_fast_processor = use_fast
        self._device = device
