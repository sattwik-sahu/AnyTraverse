import torch
import numpy as np
from numpy import typing as npt
from anytraverse import typing as anyt
from anytraverse.utils.base.map_pooling import UncertaintyPooler
from anytraverse.utils.trav_pref import get_weights
from typing_extensions import override


class ProbabilisticUncertaintyPooler(UncertaintyPooler):
    @staticmethod
    @override
    def pool(
        maps: list[anyt.PromptAttentionMap],
        traversability_preferences: anyt.TraversabilityPreferences,
    ) -> anyt.UncertaintyMap:
        return (1 - torch.stack(maps, dim=0)).prod(dim=0).squeeze(0)


class NormalizedEntropyUncertaintyPooler(UncertaintyPooler):
    @staticmethod
    @override
    def pool(
        maps: list[anyt.PromptAttentionMap],
        traversability_preferences: anyt.TraversabilityPreferences,
    ) -> anyt.UncertaintyMap:
        masks = torch.stack(maps, dim=0)
        weights: torch.Tensor = masks.max(dim=0).values
        probas: torch.Tensor = masks.softmax(dim=0)
        entropy: torch.Tensor = torch.sum(-probas * torch.log(probas), dim=0).squeeze(0)
        norm_entropy: torch.Tensor = entropy / np.log(masks.shape[0])
        return weights * norm_entropy


class InverseMaxProbabilityUncertaintyPooler(UncertaintyPooler):
    @staticmethod
    @override
    def pool(
        maps: list[anyt.PromptAttentionMap],
        traversability_preferences: anyt.TraversabilityPreferences,
    ) -> anyt.UncertaintyMap:
        return 1 - torch.stack(maps).max(dim=0).values
