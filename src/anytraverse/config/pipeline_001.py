from dataclasses import dataclass
from typing import List, Literal
from anytraverse.config.utils import (
    ScoringConfig,
    CameraConfig,
    WeightedPrompt,
    PlaneFittingConfig,
)


@dataclass
class PipelineConfig:
    prompts: List[WeightedPrompt]
    camera: CameraConfig
    device: Literal["cuda", "cpu"]
    plane_fitting: PlaneFittingConfig
    scoring: ScoringConfig
