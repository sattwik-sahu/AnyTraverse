import pytest

from anytraverse import models
from anytraverse.pipeline import AnyTraverse
from anytraverse.poolers import (
    InverseMaxProbabilityUncertaintyPooler,
    WeightedMaxTraversabilityPooler,
)
from anytraverse.presets import (
    _build_encoder,
    build_pipeline,
    build_pipeline_from_paper,
    build_pipeline_grounded_sam2,
    build_pipeline_sam3,
)

PREFS = {"road": 1.0, "bush": -0.8}


def _kwargs() -> dict:
    return {
        "init_traversability_preferences": PREFS,
        "ref_scene_similarity_threshold": 0.8,
        "roi_uncertainty_threshold": 0.3,
    }


def test_build_pipeline_wires_paper_poolers(fake_attention, fake_encoder) -> None:
    pipeline = build_pipeline(fake_attention, fake_encoder, **_kwargs())
    assert isinstance(pipeline, AnyTraverse)
    assert pipeline._traversability_pooler is WeightedMaxTraversabilityPooler
    assert pipeline._uncertainty_pooler is InverseMaxProbabilityUncertaintyPooler
    assert pipeline.traversability_preferences == PREFS


def test_from_paper_uses_clipseg_and_clip() -> None:
    pipeline = build_pipeline_from_paper(**_kwargs(), device="cpu")
    assert isinstance(pipeline._prompt_attention_mapping, models.CLIPSegAttentionMapping)
    assert isinstance(pipeline._image_encoder, models.CLIPImageEncoder)


def test_sam3_preset_and_encoder_choices() -> None:
    pipeline = build_pipeline_sam3(**_kwargs(), device="cpu")
    assert isinstance(pipeline._prompt_attention_mapping, models.SAM3AttentionMapping)
    assert isinstance(pipeline._image_encoder, models.SigLIP2ImageEncoder)
    assert isinstance(
        build_pipeline_sam3(**_kwargs(), device="cpu", encoder="dinov2")._image_encoder,
        models.DINOv2ImageEncoder,
    )
    assert isinstance(
        build_pipeline_sam3(**_kwargs(), device="cpu", encoder="clip")._image_encoder,
        models.CLIPImageEncoder,
    )


def test_grounded_preset_detector_choices() -> None:
    pipeline = build_pipeline_grounded_sam2(**_kwargs(), device="cpu")
    assert isinstance(pipeline._prompt_attention_mapping, models.GroundedSAM2AttentionMapping)
    owl = build_pipeline_grounded_sam2(**_kwargs(), device="cpu", detector="owlv2")
    assert isinstance(owl._prompt_attention_mapping, models.OWLv2SAM2AttentionMapping)
    with pytest.raises(ValueError, match="Unknown detector"):
        build_pipeline_grounded_sam2(**_kwargs(), device="cpu", detector="yolo")  # type: ignore[arg-type]


def test_unknown_encoder_raises() -> None:
    with pytest.raises(ValueError, match="Unknown encoder"):
        _build_encoder("nomic", "cpu", None)  # type: ignore[arg-type]


def test_presets_run_end_to_end(rgb_image) -> None:
    state = build_pipeline_sam3(**_kwargs(), device="cpu").step(rgb_image)
    assert state.traversability_map.shape == rgb_image.size[::-1]
    assert state.traversal_state.name in {"OK", "UNKNOWN_SCENE", "UNKNOWN_OBJECT"}
    # step() runs under inference mode so no autograd graph is built on the robot.
    assert not state.traversability_map.requires_grad
    assert not state.image_encoding.requires_grad
