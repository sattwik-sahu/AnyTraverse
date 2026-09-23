"""Poolers that combine per-prompt attention maps into traversability and uncertainty maps."""

import math
from typing import override

import torch

from anytraverse import typing as anyt
from anytraverse.interfaces import TraversabilityPooler, UncertaintyPooler
from anytraverse.preferences import get_weights


def _stack(maps: list[anyt.PromptAttentionMap]) -> torch.Tensor:
    """Stacks ``N`` maps of shape ``(H, W)`` into an ``(N, H, W)`` tensor."""
    if not maps:
        raise ValueError("At least one attention map is required")
    return torch.stack(maps, dim=0)


def _weights_like(
    traversability_preferences: anyt.TraversabilityPreferences, like: torch.Tensor
) -> torch.Tensor:
    """Returns the weights as an ``(N, 1, 1)`` tensor on the same device/dtype as ``like``."""
    weights = get_weights(traversability_preferences)
    if len(weights) != like.shape[0]:
        raise ValueError(
            f"Got {like.shape[0]} attention maps but {len(weights)} traversability preferences"
        )
    return torch.tensor(weights, device=like.device, dtype=like.dtype).reshape(-1, 1, 1)


class WeightedMaxTraversabilityPooler(TraversabilityPooler):
    """
    Traversability pooler used in the paper.

    For each pixel, the prompt with the largest absolute weighted score wins,
    and its (signed) weighted score is clipped to ``[0, 1]``.
    """

    @staticmethod
    @override
    def pool(
        maps: list[anyt.PromptAttentionMap],
        traversability_preferences: anyt.TraversabilityPreferences,
    ) -> anyt.TraversabilityMap:
        stacked = _stack(maps)
        weighted = stacked * _weights_like(traversability_preferences, stacked)
        winner = weighted.abs().argmax(dim=0, keepdim=True)
        return weighted.gather(dim=0, index=winner).squeeze(0).clip(0.0, 1.0)


class ProbabilisticTraversabilityPooler(TraversabilityPooler):
    """
    Treats prompt scores as independent probabilities.

    Traversability is the probability that at least one positive prompt is
    present and no negative prompt is present::

        (1 - prod(1 - w_i * m_i for positive i)) * prod(1 + w_j * m_j for negative j)

    Prompts with zero weight are ignored. If there are no positive prompts the
    first factor is ``1``; if there are no negative prompts the second is ``1``.
    """

    @staticmethod
    @override
    def pool(
        maps: list[anyt.PromptAttentionMap],
        traversability_preferences: anyt.TraversabilityPreferences,
    ) -> anyt.TraversabilityMap:
        stacked = _stack(maps)
        weights = _weights_like(traversability_preferences, stacked)
        weighted = stacked * weights
        positive = weights.ravel() > 0
        negative = weights.ravel() < 0
        ones = torch.ones_like(stacked[0])
        p_positive = 1 - torch.prod(1 - weighted[positive], dim=0) if positive.any() else ones
        p_no_negative = torch.prod(1 + weighted[negative], dim=0) if negative.any() else ones
        return (p_positive * p_no_negative).clip(0.0, 1.0)


class InverseMaxProbabilityUncertaintyPooler(UncertaintyPooler):
    """
    Uncertainty pooler used in the paper: ``1 - max_i m_i``.

    A pixel is uncertain when no prompt matches it strongly.
    """

    @staticmethod
    @override
    def pool(
        maps: list[anyt.PromptAttentionMap],
        traversability_preferences: anyt.TraversabilityPreferences,
    ) -> anyt.UncertaintyMap:
        return 1 - _stack(maps).max(dim=0).values


class ProbabilisticUncertaintyPooler(UncertaintyPooler):
    """Probability that no prompt is present: ``prod_i (1 - m_i)``."""

    @staticmethod
    @override
    def pool(
        maps: list[anyt.PromptAttentionMap],
        traversability_preferences: anyt.TraversabilityPreferences,
    ) -> anyt.UncertaintyMap:
        return torch.prod(1 - _stack(maps), dim=0)


class NormalizedEntropyUncertaintyPooler(UncertaintyPooler):
    """
    Entropy of the softmax over prompts, normalized to ``[0, 1]`` and scaled by
    the strongest prompt score so that empty regions do not look uncertain.

    With a single prompt the entropy is undefined and ``0`` is returned.
    """

    @staticmethod
    @override
    def pool(
        maps: list[anyt.PromptAttentionMap],
        traversability_preferences: anyt.TraversabilityPreferences,
    ) -> anyt.UncertaintyMap:
        stacked = _stack(maps)
        if stacked.shape[0] == 1:
            return torch.zeros_like(stacked[0])
        strength = stacked.max(dim=0).values
        log_probas = stacked.log_softmax(dim=0)
        entropy = -(log_probas.exp() * log_probas).sum(dim=0)
        return strength * entropy / math.log(stacked.shape[0])


__all__ = [
    "InverseMaxProbabilityUncertaintyPooler",
    "NormalizedEntropyUncertaintyPooler",
    "ProbabilisticTraversabilityPooler",
    "ProbabilisticUncertaintyPooler",
    "WeightedMaxTraversabilityPooler",
]
