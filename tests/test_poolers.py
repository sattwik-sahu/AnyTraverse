import math

import pytest
import torch

from anytraverse.poolers import (
    InverseMaxProbabilityUncertaintyPooler,
    NormalizedEntropyUncertaintyPooler,
    ProbabilisticTraversabilityPooler,
    ProbabilisticUncertaintyPooler,
    WeightedMaxTraversabilityPooler,
)
from tests.conftest import HEIGHT, WIDTH

ALL_POOLERS = [
    WeightedMaxTraversabilityPooler,
    ProbabilisticTraversabilityPooler,
    InverseMaxProbabilityUncertaintyPooler,
    ProbabilisticUncertaintyPooler,
    NormalizedEntropyUncertaintyPooler,
]


@pytest.mark.parametrize("pooler", ALL_POOLERS)
def test_shape_and_range(pooler, maps, prefs) -> None:
    out = pooler.pool(maps, prefs)
    assert out.shape == (HEIGHT, WIDTH)
    assert out.min() >= 0.0 and out.max() <= 1.0


@pytest.mark.parametrize("pooler", ALL_POOLERS)
def test_empty_maps_raise(pooler, prefs) -> None:
    with pytest.raises(ValueError, match="At least one"):
        pooler.pool([], prefs)


@pytest.mark.parametrize(
    "pooler", [WeightedMaxTraversabilityPooler, ProbabilisticTraversabilityPooler]
)
def test_mismatched_weights_raise(pooler, maps) -> None:
    with pytest.raises(ValueError, match="attention maps but"):
        pooler.pool(maps, {"road": 1.0})


class TestWeightedMax:
    def test_regions(self, maps, prefs) -> None:
        out = WeightedMaxTraversabilityPooler.pool(maps, prefs)
        # Bottom half: road dominates -> 0.9 * 1.0
        assert out[-1, -1].item() == pytest.approx(0.9)
        # Top-left: bush dominates -> -0.64 clipped to 0
        assert out[0, 0].item() == 0.0
        # Top-right: only rock -> 0.2 * 0.45
        assert out[0, -1].item() == pytest.approx(0.09)

    def test_single_prompt(self, maps) -> None:
        out = WeightedMaxTraversabilityPooler.pool(maps[:1], {"road": 0.5})
        torch.testing.assert_close(out, maps[0] * 0.5)


class TestProbabilisticTraversability:
    def test_formula(self, maps, prefs) -> None:
        out = ProbabilisticTraversabilityPooler.pool(maps, prefs)
        road, bush, rock = maps
        expected = (1 - (1 - road * 1.0) * (1 - rock * 0.45)) * (1 + bush * -0.8)
        torch.testing.assert_close(out, expected.clip(0, 1))

    def test_only_positive_weights(self, maps) -> None:
        out = ProbabilisticTraversabilityPooler.pool(maps[:1], {"road": 1.0})
        torch.testing.assert_close(out, maps[0])

    def test_only_negative_weights(self, maps) -> None:
        out = ProbabilisticTraversabilityPooler.pool(maps[1:2], {"bush": -1.0})
        torch.testing.assert_close(out, 1 - maps[1])

    def test_zero_weights_are_ignored(self, maps) -> None:
        out = ProbabilisticTraversabilityPooler.pool(maps, {"a": 0.0, "b": 0.0, "c": 0.0})
        torch.testing.assert_close(out, torch.ones_like(maps[0]))


class TestUncertainty:
    def test_inverse_max(self, maps, prefs) -> None:
        out = InverseMaxProbabilityUncertaintyPooler.pool(maps, prefs)
        torch.testing.assert_close(out, 1 - torch.stack(maps).max(0).values)

    def test_probabilistic(self, maps, prefs) -> None:
        out = ProbabilisticUncertaintyPooler.pool(maps, prefs)
        torch.testing.assert_close(out, torch.prod(1 - torch.stack(maps), dim=0))

    def test_entropy_is_zero_for_single_prompt(self, maps) -> None:
        out = NormalizedEntropyUncertaintyPooler.pool(maps[:1], {"road": 1.0})
        assert out.abs().max() == 0.0

    def test_entropy_max_when_prompts_tie(self) -> None:
        tie = [torch.full((2, 2), 0.7)] * 3
        out = NormalizedEntropyUncertaintyPooler.pool(tie, {"a": 1.0, "b": 1.0, "c": 1.0})
        # Uniform softmax -> normalized entropy 1, scaled by max score 0.7
        torch.testing.assert_close(out, torch.full((2, 2), 0.7))

    def test_entropy_normalization_constant(self) -> None:
        m = [torch.tensor([[1.0]]), torch.tensor([[0.0]])]
        out = NormalizedEntropyUncertaintyPooler.pool(m, {"a": 1.0, "b": 1.0})
        p = torch.tensor([1.0, 0.0]).softmax(0)
        expected = -(p * p.log()).sum() / math.log(2)
        assert out.item() == pytest.approx(expected.item(), rel=1e-5)
