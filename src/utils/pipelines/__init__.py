from abc import ABC, abstractmethod
from PIL.Image import Image
import torch


class CLIPSegOffnavPipeline(ABC):
    _name: str
    
    def __init__(self, name: str = "Pipeline", *args, **kwargs) -> None:
        self._name = name

    @abstractmethod
    def __call__(self, image: Image) -> torch.Tensor:
        pass
