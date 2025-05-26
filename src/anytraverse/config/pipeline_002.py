from dataclasses import dataclass
from typing import List, Literal, Optional
from anytraverse.utils.models.clipseg.pooler import CLIPSegMaskPooler
import torch

from anytraverse.config.utils import (
    CameraConfig,
    HeightScoringConfig,
    PlaneFittingConfig,
    WeightedPrompt,
)


@dataclass
class PipelineConfig:
    prompts: List[WeightedPrompt]
    mask_pooler: CLIPSegMaskPooler
    device: Literal["cuda", "cpu", "mps"] | torch.device
    camera: CameraConfig
    plane_fitting: Optional[PlaneFittingConfig] = None
    height_scoring: Optional[HeightScoringConfig] = None
    height_score: bool = True
