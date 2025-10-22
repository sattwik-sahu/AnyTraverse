from dataclasses import dataclass
from typing import Tuple

from anytraverse.utils.helpers.plane_fit import PlaneFitter


@dataclass
class PlaneFittingConfig:
    fitter: PlaneFitter
    trav_thresh: float


@dataclass
class HeightScoringConfig:
    z_thresh: Tuple[float, float]
    alpha: Tuple[float, float]


@dataclass
class ScoringConfig:
    height: HeightScoringConfig
    beta: float


@dataclass
class CameraConfig:
    fx: float
    fy: float
    cx: float
    cy: float


WeightedPrompt = Tuple[str, float]
