"""
AnyTraverse: offroad traversability with a VLM and a human operator in the loop.

Quick start::

    from anytraverse import build_pipeline_from_paper

    pipeline = build_pipeline_from_paper(
        init_traversability_preferences={"road": 1.0, "bush": -0.8, "rock": 0.45},
        ref_scene_similarity_threshold=0.8,
        roi_uncertainty_threshold=0.3,
    )
    state = pipeline.step(image)
"""

from importlib.metadata import PackageNotFoundError, version

from anytraverse import typing
from anytraverse.device import get_default_device, get_default_dtype
from anytraverse.history import EncodingHistory
from anytraverse.interfaces import (
    History,
    ImageEncoder,
    PromptAttentionMapping,
    PromptAttentionMapPooler,
    TraversabilityPooler,
    UncertaintyPooler,
)
from anytraverse.pipeline import AnyTraverse
from anytraverse.poolers import (
    InverseMaxProbabilityUncertaintyPooler,
    NormalizedEntropyUncertaintyPooler,
    ProbabilisticTraversabilityPooler,
    ProbabilisticUncertaintyPooler,
    WeightedMaxTraversabilityPooler,
)
from anytraverse.preferences import parse_trav_pref_syntax
from anytraverse.presets import (
    build_pipeline,
    build_pipeline_from_paper,
    build_pipeline_grounded_sam2,
    build_pipeline_sam3,
)
from anytraverse.roi import RegionOfInterest
from anytraverse.state import AnyTraverseState, Threshold, TraversalState

try:
    __version__ = version("anytraverse")
except PackageNotFoundError:  # pragma: no cover - only when running from a bare checkout
    __version__ = "0.0.0"

__all__ = [
    "AnyTraverse",
    "AnyTraverseState",
    "EncodingHistory",
    "History",
    "ImageEncoder",
    "InverseMaxProbabilityUncertaintyPooler",
    "NormalizedEntropyUncertaintyPooler",
    "ProbabilisticTraversabilityPooler",
    "ProbabilisticUncertaintyPooler",
    "PromptAttentionMapPooler",
    "PromptAttentionMapping",
    "RegionOfInterest",
    "Threshold",
    "TraversabilityPooler",
    "TraversalState",
    "UncertaintyPooler",
    "WeightedMaxTraversabilityPooler",
    "__version__",
    "build_pipeline",
    "build_pipeline_from_paper",
    "build_pipeline_grounded_sam2",
    "build_pipeline_sam3",
    "get_default_device",
    "get_default_dtype",
    "parse_trav_pref_syntax",
    "typing",
]
