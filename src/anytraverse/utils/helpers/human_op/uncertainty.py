from anytraverse.utils.metrics.roi import ROI_Checker
import torch
from abc import ABC, abstractmethod
from typing_extensions import override
import numpy as np


class UncertaintyChecker(ABC):
    _roi: ROI_Checker
    _thresh: float | None

    def __init__(self, roi: ROI_Checker, unc_thresh: float | None = None) -> None:
        self._roi = roi
        self._thresh = unc_thresh

    @property
    def thresh(self) -> float | None:
        return self._thresh

    @thresh.setter
    def thresh(self, unc_thresh: float | None) -> None:
        if unc_thresh is not None and (unc_thresh < 0 or unc_thresh > 1):
            raise ValueError(
                "`unc_thresh` should be `None` or a number between 0 and 1"
            )

        self.thresh = unc_thresh

    @abstractmethod
    def _get_uncertainty_mask(self, masks: torch.Tensor) -> torch.Tensor:
        pass

    def roi_uncertainty(self, masks: torch.Tensor) -> tuple[torch.Tensor, float]:
        """
        Calculates the uncertainty of detection in the ROI
        defined in the `ROI_Checker` attribute, in the given
        object segmentation masks. Takes in `n` masks, of height `h`
        and width `w`, and checks how much of the ROI does not belong
        to any of the objects being detected by the masks.

        Tells how "unknown" the ROI is. A pixel is completely unknown
        if none of the masks convey any information about it.

        Args:
            masks (torch.Tensor): The masks to help detect the ROI
                uncertainty. Dim: `(n, h, w)`

        Returns:
            (torch.Tensor, float): Returns a tuple of a `torch.Tensor` and a `float`
                - The uncertainty mask, where `1.0` means most uncertain and `0.0` means least uncertain.
                - A single number between `[0.0, 1.0]` telling the amount of uncertainty in the ROI of the masks.
        """
        masks_: torch.Tensor = masks if self.thresh is None else masks > self.thresh
        unc_mask: torch.Tensor = self._get_uncertainty_mask(masks=masks_)
        return unc_mask, self._roi.trav_area(mask=unc_mask)


class ProbabilisticUncertaintyChecker(UncertaintyChecker):
    """
    Performs probabilistic checking on the uncertainty
    of the ROI in given traversability/detection masks.
    """

    def __init__(self, roi: ROI_Checker) -> None:
        super().__init__(roi=roi)

    @override
    def _get_uncertainty_mask(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Calculates the probabilistic uncertainty in the ROI of the masks.
        Subtracts each mask from 1 to get the undetected-ness of different
        pixels. Then gets the product of corresponding pixel values of the
        undetected-ness masks.

        Args:
            masks (torch.Tensor): The masks to help detect the ROI
                uncertainty. Dim: `(n, h, w)`

        Returns:
            float: A single number between [0, 1] telling the amount of
                uncertainty in the ROI of the masks.
        """
        undet_masks: torch.Tensor = 1 - masks
        uncertainty_mask: torch.Tensor = undet_masks.prod(dim=0).squeeze(0)
        return uncertainty_mask


class NormalizedEntropyUncertaintyChecker(UncertaintyChecker):
    """
    Uncertainty checker using normalized entropy.
    """

    def __init__(self, roi: ROI_Checker) -> None:
        super().__init__(roi=roi)

    @override
    def _get_uncertainty_mask(self, masks: torch.Tensor) -> torch.Tensor:
        weights: torch.Tensor = masks.max(dim=0).values
        probas: torch.Tensor = masks.softmax(dim=0)
        entropy: torch.Tensor = torch.sum(-probas * torch.log(probas), dim=0).squeeze(0)
        norm_entropy: torch.Tensor = entropy / np.log(masks.shape[0])
        return weights * norm_entropy


class InvMaxProbaUncertaintyChecker(UncertaintyChecker):
    """
    Uncertainty checker that works on the principle that:
    uncertainty = 1 - max(probas of all prompts).

    This method produces very similar results to the
    `NormalizedEntropyUncertaintyChecker` somehow.

    TODO See why are the results so similar?
    """

    def __init__(self, roi: ROI_Checker) -> None:
        super().__init__(roi=roi)

    @override
    def _get_uncertainty_mask(self, masks: torch.Tensor) -> torch.Tensor:
        return 1 - masks.max(dim=0).values.squeeze()
