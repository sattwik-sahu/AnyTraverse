from abc import ABC, abstractmethod
from typing import NamedTuple
from PIL.Image import Image
import torch

from utils.pipelines.height_scoring import HeightScoringOutput


class PipelineOutput(NamedTuple):
    trav_masks: torch.Tensor
    pooled_mask: torch.Tensor
    height_scores: HeightScoringOutput | None
    output: torch.Tensor


class CLIPSegOffnavPipeline(ABC):
    _name: str

    def __init__(self, name: str = "Pipeline", *args, **kwargs) -> None:
        self._name = name

    @abstractmethod
    def __call__(self, image: Image) -> PipelineOutput:
        pass
