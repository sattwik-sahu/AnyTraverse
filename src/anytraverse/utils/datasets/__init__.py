from abc import ABC, abstractmethod
from typing import TypeVar
import numpy as np
from PIL.Image import Image
from numpy import typing as npt

TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


class Processor[TInput, TOutput](ABC):
    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def __call__(self, x: TInput) -> TOutput:
        pass


class ImageProcessor(Processor[Image, npt.NDArray]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
