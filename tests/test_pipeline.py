import pytest
import torch

from anytraverse.pipeline import AnyTraverse
from anytraverse.poolers import (
    InverseMaxProbabilityUncertaintyPooler,
    WeightedMaxTraversabilityPooler,
)
from anytraverse.roi import RegionOfInterest
from anytraverse.state import AnyTraverseState, Threshold, TraversalState
from tests.conftest import HEIGHT, WIDTH, FakeAttentionMapping, FakeEncoder, unit

SIM_THRESHOLD = 0.8
UNC_THRESHOLD = 0.3


def make_pipeline(attention: FakeAttentionMapping, encoder: FakeEncoder, **kwargs) -> AnyTraverse:
    return AnyTraverse(
        prompt_attention_mapping=attention,
        image_encoder=encoder,
        traversability_pooler=WeightedMaxTraversabilityPooler,
        uncertainty_pooler=InverseMaxProbabilityUncertaintyPooler,
        init_traversability_preferences={"road": 1.0, "bush": -0.8, "rock": 0.45},
        roi=RegionOfInterest(x_bounds=(0.25, 0.75), y_bounds=(0.5, 1.0)),
        threshold=Threshold(ref_scene_similarity=SIM_THRESHOLD, roi_uncertainty=UNC_THRESHOLD),
        **kwargs,
    )


class TestFirstStep:
    def test_registers_reference_scene_and_returns_ok(
        self, fake_attention, fake_encoder, rgb_image
    ):
        pipeline = make_pipeline(fake_attention, fake_encoder)
        state = pipeline.step(rgb_image)

        assert isinstance(state, AnyTraverseState)
        assert state.traversal_state is TraversalState.OK
        assert state.ref_scene_similarity == pytest.approx(1.0)
        assert len(pipeline.history) == 1
        assert fake_attention.calls == [["road", "bush", "rock"]]

    def test_output_shapes(self, fake_attention, fake_encoder, rgb_array):
        state = make_pipeline(fake_attention, fake_encoder).step(rgb_array)
        assert state.traversability_map.shape == (HEIGHT, WIDTH)
        assert state.uncertainty_map.shape == (HEIGHT, WIDTH)
        assert len(state.attention_maps) == 3
        assert state.roi_bbox == ((8, 12), (24, 23))
        assert state.traversability_map_roi.shape == (12, 17)
        assert state.roi_traversability == pytest.approx(0.9)
        assert state.roi_uncertainty == pytest.approx(0.1)
        assert state.traversability_preferences == {"road": 1.0, "bush": -0.8, "rock": 0.45}

    def test_returned_preferences_are_a_copy(self, fake_attention, fake_encoder, rgb_image):
        pipeline = make_pipeline(fake_attention, fake_encoder)
        state = pipeline.step(rgb_image)
        state.traversability_preferences["road"] = 0.0
        assert pipeline.traversability_preferences["road"] == 1.0


class TestDecisions:
    def test_unknown_object_when_roi_uncertain(self, fake_encoder, rgb_image):
        attention = FakeAttentionMapping(default=0.2)  # 1 - max = 0.8 > threshold
        state = make_pipeline(attention, fake_encoder).step(rgb_image)
        assert state.traversal_state is TraversalState.UNKNOWN_OBJECT

    def test_unknown_scene_when_nothing_similar(self, fake_attention, rgb_image):
        encoder = FakeEncoder([unit(1.0), unit(0.0, 1.0)])
        pipeline = make_pipeline(fake_attention, encoder)
        pipeline.step(rgb_image)
        state = pipeline.step(rgb_image)
        assert state.traversal_state is TraversalState.UNKNOWN_SCENE
        assert state.ref_scene_similarity == pytest.approx(0.0)

    def test_scene_check_takes_priority_over_roi_uncertainty(self, rgb_image):
        encoder = FakeEncoder([unit(1.0), unit(0.0, 1.0)])
        pipeline = make_pipeline(FakeAttentionMapping(default=0.2), encoder)
        pipeline.step(rgb_image)
        assert pipeline.step(rgb_image).traversal_state is TraversalState.UNKNOWN_SCENE

    def test_history_hit_restores_preferences_and_reference(self, fake_attention, rgb_image):
        a, b = unit(1.0), unit(0.0, 1.0)
        encoder = FakeEncoder([a, b, b, a])
        pipeline = make_pipeline(fake_attention, encoder)

        pipeline.step(rgb_image)  # register scene a
        pipeline.step(rgb_image)  # scene b -> unknown scene
        pipeline.human_call("mud: -0.5")  # operator registers b with new prefs
        assert pipeline.step(rgb_image).traversal_state is TraversalState.OK  # b is reference

        state = pipeline.step(rgb_image)  # back to a: found in history
        assert state.traversal_state is TraversalState.OK
        assert state.ref_scene_similarity < SIM_THRESHOLD
        # Preferences from the matched history element are merged in.
        assert pipeline.traversability_preferences == {
            "road": 1.0,
            "bush": -0.8,
            "rock": 0.45,
            "mud": -0.5,
        }
        # Reference is now scene a again, so the next frame of a is fully similar.
        encoder.queue.append(a)
        assert pipeline.step(rgb_image).ref_scene_similarity == pytest.approx(1.0)

    def test_history_size_limits_memory(self, fake_attention, rgb_image):
        encoder = FakeEncoder([unit(1.0), unit(0.0, 1.0)])
        pipeline = make_pipeline(fake_attention, encoder, history_size=1)
        pipeline.step(rgb_image)
        pipeline.step(rgb_image)
        pipeline.register_scene()
        assert len(pipeline.history) == 1


class TestHumanCall:
    def test_updates_preferences_and_prompts_used(self, fake_attention, fake_encoder, rgb_image):
        pipeline = make_pipeline(fake_attention, fake_encoder)
        pipeline.step(rgb_image)
        element = pipeline.human_call("mud: -0.5; road: 0.9")
        assert element[1] == {"road": 0.9, "bush": -0.8, "rock": 0.45, "mud": -0.5}
        pipeline.step(rgb_image)
        assert fake_attention.calls[-1] == ["road", "bush", "rock", "mud"]
        assert len(pipeline.history) == 2

    def test_empty_input_only_registers_scene(self, fake_attention, fake_encoder, rgb_image):
        pipeline = make_pipeline(fake_attention, fake_encoder)
        pipeline.step(rgb_image)
        before = pipeline.traversability_preferences
        pipeline.human_call("   ")
        assert pipeline.traversability_preferences == before
        assert len(pipeline.history) == 2

    def test_register_before_step_raises(self, fake_attention, fake_encoder):
        with pytest.raises(RuntimeError, match="step\\(\\)"):
            make_pipeline(fake_attention, fake_encoder).register_scene()

    def test_invalid_weight_from_operator_raises(self, fake_attention, fake_encoder, rgb_image):
        pipeline = make_pipeline(fake_attention, fake_encoder)
        pipeline.step(rgb_image)
        with pytest.raises(ValueError, match="must be in"):
            pipeline.human_call("mud: 5")


class TestPreferencesProperty:
    def test_setter_validates(self, fake_attention, fake_encoder):
        pipeline = make_pipeline(fake_attention, fake_encoder)
        pipeline.traversability_preferences = {"grass": 0.5}
        assert pipeline.traversability_preferences == {"grass": 0.5}
        with pytest.raises(ValueError):
            pipeline.traversability_preferences = {"grass": 2.0}

    def test_init_validates(self, fake_attention, fake_encoder):
        with pytest.raises(ValueError):
            make_pipeline(fake_attention, fake_encoder).__class__(
                prompt_attention_mapping=fake_attention,
                image_encoder=fake_encoder,
                traversability_pooler=WeightedMaxTraversabilityPooler,
                uncertainty_pooler=InverseMaxProbabilityUncertaintyPooler,
                init_traversability_preferences={},
                roi=RegionOfInterest((0.0, 1.0), (0.0, 1.0)),
                threshold=Threshold(0.8, 0.3),
            )

    def test_accessors(self, fake_attention, fake_encoder, rgb_image):
        pipeline = make_pipeline(fake_attention, fake_encoder)
        assert pipeline.threshold == Threshold(SIM_THRESHOLD, UNC_THRESHOLD)
        assert pipeline.traversal_state is TraversalState.OK
        pipeline.step(rgb_image)
        assert pipeline.traversal_state is TraversalState.OK


def test_custom_similarity_function(fake_attention, rgb_image):
    calls = []

    def dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        calls.append((a.shape, b.shape))
        return (a * b).sum(dim=-1)

    encoder = FakeEncoder([unit(1.0), unit(0.0, 1.0)])
    pipeline = make_pipeline(fake_attention, encoder, similarity_func=dot)
    pipeline.step(rgb_image)
    pipeline.step(rgb_image)
    # One call for the reference check and one vectorized history lookup per step.
    assert len(calls) == 3
