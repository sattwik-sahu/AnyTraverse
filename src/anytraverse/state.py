"""Data classes describing the pipeline's configuration and per-step output."""

from dataclasses import dataclass, fields
from enum import Enum

import torch

from anytraverse import typing as anyt


class TraversalState(Enum):
    """The decision taken by the pipeline after processing a frame."""

    OK = "ok"
    """Safe to continue; no operator call needed."""

    UNKNOWN_SCENE = "unknown_scene"
    """The scene differs from the reference scene and nothing similar is in the history."""

    UNKNOWN_OBJECT = "unknown_object"
    """The region of interest is too uncertain; an unknown object may be present."""


@dataclass(frozen=True, slots=True)
class Threshold:
    """
    Thresholds controlling when a human operator call is triggered.

    Attributes:
        ref_scene_similarity (float): Minimum similarity between the current
            and reference scene encodings for the scene to be considered known.
        roi_uncertainty (float): Maximum mean uncertainty allowed in the region
            of interest before an unknown object is flagged.
    """

    ref_scene_similarity: float
    roi_uncertainty: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.roi_uncertainty <= 1.0:
            raise ValueError("roi_uncertainty must be in [0, 1]")


@dataclass(slots=True)
class AnyTraverseState:
    """
    Everything the pipeline computed for one frame.

    All maps are ``torch.Tensor`` objects of shape ``(H, W)`` matching the input
    image, unless noted otherwise.
    """

    image_encoding: anyt.Encoding
    """Encoding of the input image, shape ``(1, D)``."""

    attention_maps: list[anyt.PromptAttentionMap]
    """One attention map per prompt, in the order of ``traversability_preferences``."""

    traversability_map: anyt.TraversabilityMap
    """Per-pixel traversability from ``0`` (untraversable) to ``1`` (traversable)."""

    uncertainty_map: anyt.UncertaintyMap
    """Per-pixel uncertainty from ``0`` (certain) to ``1`` (uncertain)."""

    traversability_map_roi: anyt.TraversabilityMap
    """The region of interest cropped from ``traversability_map``."""

    uncertainty_map_roi: anyt.UncertaintyMap
    """The region of interest cropped from ``uncertainty_map``."""

    ref_scene_similarity: float
    """Similarity between the current scene and the reference scene."""

    traversability_preferences: anyt.TraversabilityPreferences
    """The prompts and weights that were used for this frame."""

    roi_bbox: tuple[tuple[int, int], tuple[int, int]]
    """The region of interest as ``((x_start, y_start), (x_end, y_end))`` in pixels."""

    roi_uncertainty: float
    """Mean uncertainty inside the region of interest."""

    roi_traversability: float
    """Mean traversability inside the region of interest."""

    traversal_state: TraversalState
    """Whether it is safe to continue or an operator call is required."""

    def to(self, device: str | torch.device) -> "AnyTraverseState":
        """
        Moves all tensors to a device, e.g. ``"cpu"`` before publishing over ROS.

        Args:
            device (str | torch.device): The target device.

        Returns:
            AnyTraverseState: A new state with tensors on ``device``.
        """
        moved: dict[str, object] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                value = value.to(device)
            elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
                value = [v.to(device) for v in value]
            moved[field.name] = value
        return AnyTraverseState(**moved)  # type: ignore[arg-type]


__all__ = ["AnyTraverseState", "Threshold", "TraversalState"]
