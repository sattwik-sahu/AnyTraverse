from utils.metrics.roi import ROI_Checker
import torch
from abc import ABC, abstractmethod
from typing_extensions import override


class UncertaintyChecker(ABC):
    _roi: ROI_Checker

    def __init__(self, roi: ROI_Checker) -> None:
        self._roi = roi

    @abstractmethod
    def _get_uncertainty_mask(self, masks: torch.Tensor) -> torch.Tensor:
        pass

    def roi_uncertainty(self, masks: torch.Tensor) -> float:
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
            float: A single number between [0, 1] telling the amount of
                uncertainty in the ROI of the masks.
        """
        return self._roi.trav_area(mask=self._get_uncertainty_mask(masks=masks))


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
