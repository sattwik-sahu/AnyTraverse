from abc import ABC, abstractmethod
import torch
import numpy as np
from typing_extensions import override
from typing import List, Literal


class MaskPooler(ABC):
    @staticmethod
    @abstractmethod
    def pool(masks: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        pass


class SumPooler(MaskPooler):
    @staticmethod
    @override
    def pool(masks: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return masks.squeeze(1).sum(dim=0)


class MaxPooler(MaskPooler):
    @staticmethod
    @override
    def pool(masks: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return masks.squeeze(1).max(dim=0).values


class MeanPooler(MaskPooler):
    @staticmethod
    @override
    def pool(masks: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return masks.squeeze(1).mean(dim=0)


class WeightedSumPooler(MaskPooler):
    @staticmethod
    @override
    def pool(
        masks: torch.Tensor, weights: List[float], *args, **kwargs
    ) -> torch.Tensor:
        masks_ = masks.squeeze(1)
        pooled_mask: torch.Tensor = torch.zeros_like(masks[0])
        for mask, weight in zip(masks_, weights):
            pooled_mask += mask * weight
        pooled_mask.clip(min=0.0, max=1.0)
        return pooled_mask.squeeze(0)


class WeightedMaxPooler(MaskPooler):
    @staticmethod
    @override
    def pool(
        masks: torch.Tensor, weights: List[float], *args, **kwargs
    ) -> torch.Tensor:
        masks_ = masks.squeeze(1)
        # print(masks_.shape, len(weights))
        pooled_list = []
        pooled_mask: torch.Tensor = torch.zeros_like(masks[0])
        for mask, weight in zip(masks_, weights):
            pooled_list.append(mask * weight)

        # pooled_mask = torch.stack(pooled_list).max(dim=0).values
        stacked = torch.stack(pooled_list)
        abs_stacked = stacked.abs()
        _, indices = abs_stacked.max(dim=0)
        pooled_mask = stacked.gather(dim=0, index=indices.unsqueeze(0)).squeeze(0)

        # print(pooled_mask.shape)
        pooled_mask.clip(min=0.0, max=1.0)
        # return pooled_mask.squeeze(0)
        return pooled_mask


class ProbabilisticPooler(MaskPooler):
    @staticmethod
    @override
    def pool(
        masks: torch.Tensor,
        weights: List[float],
        device: Literal["cpu", "cuda", "mps"] = "cpu",
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Probabilistic Pooler.

        Args:
            masks (torch.Tensor): The output masks from CLIPSeg.
                Dimensions: `(N, H, W)`
            weights (List[float]): The list of weights for each of the masks.
                - Positive values signify the probability that the pixel is
                    traversible, given it belongs to the i-th class (prompt).
                    Size: `N`
                - Negative values signify the negative probability that the
                    pixel is non-traversible, given it belongs to the i-th
                    class (prompt). Size: `N`
            device (torch.device): The device to perform the pooling
                calculations on.

        Returns:
            torch.Tensor: The probabilistically pooled attention mask,
                highlighting the traversible regions.
                Dimension: `(H, W)`
        """
        _device: torch.device = torch.device(device=device)
        weights_arr = np.array(weights).astype(np.float32)
        weights_tensor = torch.tensor(weights_arr).reshape(-1, 1, 1).to(device=_device)
        pos_inx, neg_inx = weights_arr > 0, weights_arr < 0
        z = weights_tensor * masks.squeeze(1)
        proba_trav = (1 - torch.prod(1 - z[pos_inx.flatten()], dim=0)) * torch.prod(  # type: ignore
            1 + z[neg_inx.flatten()],  # type: ignore
            dim=0,
            dtype=torch.float32,
        )

        return proba_trav
