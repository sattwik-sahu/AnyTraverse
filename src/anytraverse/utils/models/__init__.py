from abc import ABC, abstractmethod
from typing import TypeVar

import torch
from PIL.Image import Image

TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


class Model[TInput, TOutput](ABC):
    _device: torch.device

    def __init__(self, device: torch.device = torch.device("cuda")) -> None:
        super().__init__()
        self._device = device

    @abstractmethod
    def __call__(self, x: TInput) -> TOutput:
        pass


ImageEmbeddingModel = Model[Image, torch.Tensor]
