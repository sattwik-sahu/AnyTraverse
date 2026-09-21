import pytest
import torch

from anytraverse.state import AnyTraverseState, Threshold, TraversalState


def test_traversal_state_members() -> None:
    assert {s.name for s in TraversalState} == {"OK", "UNKNOWN_SCENE", "UNKNOWN_OBJECT"}


def test_threshold_is_frozen_and_validated() -> None:
    threshold = Threshold(ref_scene_similarity=0.8, roi_uncertainty=0.3)
    with pytest.raises(AttributeError):
        threshold.roi_uncertainty = 0.5  # type: ignore[misc]
    with pytest.raises(ValueError, match="roi_uncertainty"):
        Threshold(ref_scene_similarity=0.8, roi_uncertainty=1.5)


def _state() -> AnyTraverseState:
    m = torch.zeros(4, 4)
    return AnyTraverseState(
        image_encoding=torch.zeros(1, 2),
        attention_maps=[m, m.clone()],
        traversability_map=m,
        uncertainty_map=m,
        traversability_map_roi=m[2:, 1:3],
        uncertainty_map_roi=m[2:, 1:3],
        ref_scene_similarity=1.0,
        traversability_preferences={"road": 1.0},
        roi_bbox=((1, 2), (2, 3)),
        roi_uncertainty=0.0,
        roi_traversability=0.0,
        traversal_state=TraversalState.OK,
    )


def test_state_to_moves_all_tensors() -> None:
    moved = _state().to("cpu")
    assert moved.image_encoding.device.type == "cpu"
    assert all(m.device.type == "cpu" for m in moved.attention_maps)
    assert moved.traversal_state is TraversalState.OK
    assert moved.roi_bbox == ((1, 2), (2, 3))
    assert moved.traversability_preferences == {"road": 1.0}


def test_state_to_handles_empty_attention_maps() -> None:
    state = _state()
    state.attention_maps = []
    assert state.to("cpu").attention_maps == []
