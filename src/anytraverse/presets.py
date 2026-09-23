"""Factory functions that assemble ready-to-run pipelines."""

from typing import Literal

import torch

from anytraverse import typing as anyt
from anytraverse.interfaces import ImageEncoder, PromptAttentionMapping
from anytraverse.pipeline import AnyTraverse
from anytraverse.poolers import (
    InverseMaxProbabilityUncertaintyPooler,
    WeightedMaxTraversabilityPooler,
)
from anytraverse.roi import RegionOfInterest
from anytraverse.state import Threshold

EncoderName = Literal["clip", "siglip2", "dinov2"]


def _build_encoder(
    name: EncoderName, device: str | torch.device | None, dtype: torch.dtype | None
) -> ImageEncoder:
    """Instantiates a scene encoder by name."""
    from anytraverse import models

    match name:
        case "clip":
            return models.CLIPImageEncoder(device=device, dtype=dtype)
        case "siglip2":
            return models.SigLIP2ImageEncoder(device=device, dtype=dtype)
        case "dinov2":
            return models.DINOv2ImageEncoder(device=device, dtype=dtype)
    raise ValueError(f"Unknown encoder {name!r}; expected 'clip', 'siglip2' or 'dinov2'")


def build_pipeline(
    prompt_attention_mapping: PromptAttentionMapping,
    image_encoder: ImageEncoder,
    init_traversability_preferences: anyt.TraversabilityPreferences,
    ref_scene_similarity_threshold: float,
    roi_uncertainty_threshold: float,
    roi_x_bounds: tuple[float, float] = (0.333, 0.667),
    roi_y_bounds: tuple[float, float] = (0.6, 0.95),
    history_size: int | None = None,
) -> AnyTraverse:
    """
    Assembles a pipeline with the poolers from the paper around any two models.

    Args:
        prompt_attention_mapping (PromptAttentionMapping): The VLM producing attention maps.
        image_encoder (ImageEncoder): The scene encoder used for the history.
        init_traversability_preferences (TraversabilityPreferences): Initial prompts and weights.
        ref_scene_similarity_threshold (float): See :class:`Threshold`.
        roi_uncertainty_threshold (float): See :class:`Threshold`.
        roi_x_bounds (tuple[float, float]): Fractional width bounds of the region of interest.
        roi_y_bounds (tuple[float, float]): Fractional height bounds of the region of interest.
        history_size (int | None): Maximum number of remembered scenes.

    Returns:
        AnyTraverse: The assembled pipeline.
    """
    return AnyTraverse(
        prompt_attention_mapping=prompt_attention_mapping,
        image_encoder=image_encoder,
        traversability_pooler=WeightedMaxTraversabilityPooler,
        uncertainty_pooler=InverseMaxProbabilityUncertaintyPooler,
        init_traversability_preferences=init_traversability_preferences,
        roi=RegionOfInterest(x_bounds=roi_x_bounds, y_bounds=roi_y_bounds),
        threshold=Threshold(
            ref_scene_similarity=ref_scene_similarity_threshold,
            roi_uncertainty=roi_uncertainty_threshold,
        ),
        history_size=history_size,
    )


def build_pipeline_from_paper(
    init_traversability_preferences: anyt.TraversabilityPreferences,
    ref_scene_similarity_threshold: float,
    roi_uncertainty_threshold: float,
    roi_x_bounds: tuple[float, float] = (0.333, 0.667),
    roi_y_bounds: tuple[float, float] = (0.6, 0.95),
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> AnyTraverse:
    """
    Builds the pipeline from the paper: CLIPSeg attention maps and CLIP ViT-B/32 scene encodings.

    Args:
        init_traversability_preferences (TraversabilityPreferences): Initial prompts and weights.
        ref_scene_similarity_threshold (float): See :class:`Threshold`.
        roi_uncertainty_threshold (float): See :class:`Threshold`.
        roi_x_bounds (tuple[float, float]): Fractional width bounds of the region of interest.
        roi_y_bounds (tuple[float, float]): Fractional height bounds of the region of interest.
        device (str | torch.device | None): Device for both models; auto-detected if ``None``.
        dtype (torch.dtype | None): Precision for both models; see :class:`HuggingFaceModel`.

    Returns:
        AnyTraverse: The assembled pipeline.
    """
    from anytraverse import models

    return build_pipeline(
        prompt_attention_mapping=models.CLIPSegAttentionMapping(device=device, dtype=dtype),
        image_encoder=models.CLIPImageEncoder(device=device, dtype=dtype),
        init_traversability_preferences=init_traversability_preferences,
        ref_scene_similarity_threshold=ref_scene_similarity_threshold,
        roi_uncertainty_threshold=roi_uncertainty_threshold,
        roi_x_bounds=roi_x_bounds,
        roi_y_bounds=roi_y_bounds,
    )


def build_pipeline_sam3(
    init_traversability_preferences: anyt.TraversabilityPreferences,
    ref_scene_similarity_threshold: float,
    roi_uncertainty_threshold: float,
    roi_x_bounds: tuple[float, float] = (0.333, 0.667),
    roi_y_bounds: tuple[float, float] = (0.6, 0.95),
    encoder: EncoderName = "siglip2",
    image_size: int | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> AnyTraverse:
    """
    Builds a pipeline with SAM 3 attention maps.

    Args:
        init_traversability_preferences (TraversabilityPreferences): Initial prompts and weights.
        ref_scene_similarity_threshold (float): See :class:`Threshold`.
        roi_uncertainty_threshold (float): See :class:`Threshold`.
        roi_x_bounds (tuple[float, float]): Fractional width bounds of the region of interest.
        roi_y_bounds (tuple[float, float]): Fractional height bounds of the region of interest.
        encoder (Literal["clip", "siglip2", "dinov2"]): Scene encoder for the history.
        image_size (int | None): SAM 3 input resolution; ``560`` for low-memory devices.
        device (str | torch.device | None): Device for all models; auto-detected if ``None``.
        dtype (torch.dtype | None): Precision for all models; see :class:`HuggingFaceModel`.

    Returns:
        AnyTraverse: The assembled pipeline.
    """
    from anytraverse import models

    return build_pipeline(
        prompt_attention_mapping=models.SAM3AttentionMapping(
            image_size=image_size, device=device, dtype=dtype
        ),
        image_encoder=_build_encoder(encoder, device, dtype),
        init_traversability_preferences=init_traversability_preferences,
        ref_scene_similarity_threshold=ref_scene_similarity_threshold,
        roi_uncertainty_threshold=roi_uncertainty_threshold,
        roi_x_bounds=roi_x_bounds,
        roi_y_bounds=roi_y_bounds,
    )


def build_pipeline_grounded_sam2(
    init_traversability_preferences: anyt.TraversabilityPreferences,
    ref_scene_similarity_threshold: float,
    roi_uncertainty_threshold: float,
    roi_x_bounds: tuple[float, float] = (0.333, 0.667),
    roi_y_bounds: tuple[float, float] = (0.6, 0.95),
    encoder: EncoderName = "siglip2",
    detector: Literal["grounding-dino", "owlv2"] = "grounding-dino",
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> AnyTraverse:
    """
    Builds a pipeline with detector + SAM 2 attention maps.

    Args:
        init_traversability_preferences (TraversabilityPreferences): Initial prompts and weights.
        ref_scene_similarity_threshold (float): See :class:`Threshold`.
        roi_uncertainty_threshold (float): See :class:`Threshold`.
        roi_x_bounds (tuple[float, float]): Fractional width bounds of the region of interest.
        roi_y_bounds (tuple[float, float]): Fractional height bounds of the region of interest.
        encoder (Literal["clip", "siglip2", "dinov2"]): Scene encoder for the history.
        detector (Literal["grounding-dino", "owlv2"]): Open-vocabulary detector to use.
        device (str | torch.device | None): Device for all models; auto-detected if ``None``.
        dtype (torch.dtype | None): Precision for all models; see :class:`HuggingFaceModel`.

    Returns:
        AnyTraverse: The assembled pipeline.
    """
    from anytraverse import models

    match detector:
        case "grounding-dino":
            mapping: PromptAttentionMapping = models.GroundedSAM2AttentionMapping(
                device=device, dtype=dtype
            )
        case "owlv2":
            mapping = models.OWLv2SAM2AttentionMapping(device=device, dtype=dtype)
        case _:
            raise ValueError(f"Unknown detector {detector!r}; expected 'grounding-dino' or 'owlv2'")

    return build_pipeline(
        prompt_attention_mapping=mapping,
        image_encoder=_build_encoder(encoder, device, dtype),
        init_traversability_preferences=init_traversability_preferences,
        ref_scene_similarity_threshold=ref_scene_similarity_threshold,
        roi_uncertainty_threshold=roi_uncertainty_threshold,
        roi_x_bounds=roi_x_bounds,
        roi_y_bounds=roi_y_bounds,
    )


__all__ = [
    "build_pipeline",
    "build_pipeline_from_paper",
    "build_pipeline_grounded_sam2",
    "build_pipeline_sam3",
]
