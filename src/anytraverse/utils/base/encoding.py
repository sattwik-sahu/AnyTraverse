from abc import ABC, abstractmethod
from typing import Any
from anytraverse import typing as anyt


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
        self, x: TInput | list[TInput], *args: Any, **kwargs: Any
    ) -> anyt.Encoding:
        pass


class ImageEncoder[TImage: anyt.Image](BaseEncoder[TImage], ABC):
    """
    Base class for an image encoder module.
    """
