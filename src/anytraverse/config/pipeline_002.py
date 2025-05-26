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
    """
    Configuration for the AnyTraverse pipeline.
    This configuration includes the model prompts, mask pooler, device settings,
    camera configuration, and optional plane fitting and height scoring configurations.

    Attributes:
        prompts (List[WeightedPrompt]): List of weighted prompts for the attention mapping model.
        mask_pooler (CLIPSegMaskPooler): Pooler for combining multiple attention masks.
        device (Literal["cuda", "cpu", "mps"] | torch.device): Device to run the model on.
        camera (CameraConfig): Configuration for the camera used in the pipeline. *Deprecated*
        plane_fitting (Optional[PlaneFittingConfig]): Configuration for plane fitting, if applicable. *Deprecated*
        height_scoring (Optional[HeightScoringConfig]): Configuration for height scoring, if applicable. *Deprecated*
        height_score (bool): Flag to determine if height scoring should be performed. *Deprecated*
    """

    prompts: List[WeightedPrompt]
    mask_pooler: type[CLIPSegMaskPooler] | CLIPSegMaskPooler
    device: Literal["cuda", "cpu", "mps"] | torch.device
    # camera: CameraConfig
    # plane_fitting: Optional[PlaneFittingConfig] = None
    # height_scoring: Optional[HeightScoringConfig] = None
    # height_score: bool = True
