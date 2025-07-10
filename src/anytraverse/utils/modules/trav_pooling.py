import torch
import numpy as np
from numpy import typing as npt
from anytraverse import typing as anyt
from anytraverse.utils.base.map_pooling import TraversabilityPooler
from anytraverse.utils.trav_pref import get_weights
from typing_extensions import override


class ProbabilisticTraversabilityPooler(TraversabilityPooler):
    @staticmethod
    @override
    def pool(
        maps: list[anyt.PromptAttentionMap],
        traversability_preferences: anyt.TraversabilityPreferences,
    ) -> anyt.TraversabilityMap:
        """
        Probabilistic traversability pooler.

        Args:
            maps (list[PromptAttentionMap]):
                The prompt attention maps for each prompt, on the image.
                Each map is a `torch.Tensor` of shape `(H, W)`, where `H`
                is the height of the image and `W` is the width.
            traversability_preferences (TraversabilityPreferences):
                The traversability preferences of the AnyTraverse pipeline.

        Returns:
            TraversabilityMap:
                A `torch.Tensor` of shape `(H, W)`, showing traversability
                of each of the pixels in the image.
        """
        device = maps[0].device
        weights = get_weights(traversability_preferences=traversability_preferences)
        weights_arr = np.array(weights)
        weights_tensor = torch.tensor(weights_arr).reshape(-1, 1, 1).to(device=device)
        pos_inx, neg_inx = weights_arr > 0, weights_arr < 0
        z = weights_tensor * torch.stack(maps, dim=0)
        proba_trav = (1 - torch.prod(1 - z[pos_inx.flatten()], dim=0)) * torch.prod(  # type: ignore
            1 + z[neg_inx.flatten()],  # type: ignore
            dim=0,
            dtype=torch.float32,
        )

        return proba_trav


class WeightedMaxTraversabilityPooler(TraversabilityPooler):
    """
    The weighted max traversability pooler discussed in the paper.
    """

    @staticmethod
    @override
    def pool(
        maps: list[anyt.PromptAttentionMap],
        traversability_preferences: anyt.TraversabilityPreferences,
    ) -> anyt.TraversabilityMap:
        # Create weighted maps
        weights = get_weights(traversability_preferences=traversability_preferences)
        weighted_maps = torch.stack(
            [weight * attn_map for attn_map, weight in zip(maps, weights)], dim=0
        )
        # Get the absolute weighted maps
        abs_weighted_maps = weighted_maps.abs()
        # For each pixel, which map has the highest absolute weighted value?
        _, max_abs_weighted_map_index = abs_weighted_maps.max(dim=0)
        # Create the pooled traversability map
        traversability_map: anyt.TraversabilityMap = (
            weighted_maps.gather(dim=0, index=max_abs_weighted_map_index.unsqueeze(0))
            .squeeze(0)
            .clip(min=0.0, max=1.0)
        )
        return traversability_map
