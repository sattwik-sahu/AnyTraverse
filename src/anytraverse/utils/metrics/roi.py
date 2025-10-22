import torch

from typing import Tuple


BboxBounds = Tuple[float, float]
BboxBoundsPixel = Tuple[int, int]


class ROI_Checker:
    """
    Region of interest (ROI) checker for output
    of the AnyTraverse traversability segmentation
    pipeline.

    Attributes:
        _x_bounds (List[float]): The relative horizontal bounds of the
            bbox of the ROI in thr format `(x_lb, x_ub)`, where both the
            bounds are in the range `[0, 1]` and `x_lb < x_ub`.
        _y_bounds (List[float]): The relative vertical bounds of the
            bbox of the ROI in thr format `(y_lb, y_ub)`, where both the
            bounds are in the range `[0, 1]` and `y_lb < y_ub`.
    """

    _x_bounds: BboxBounds
    _y_bounds: BboxBounds
    _device: torch.device

    def __init__(
        self,
        x_bounds: BboxBounds,
        y_bounds: BboxBounds,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        self._x_bounds = x_bounds
        self._y_bounds = y_bounds
        self._device = device

    @property
    def x_bounds(self) -> BboxBounds:
        return self.x_bounds

    @property
    def y_bounds(self) -> BboxBounds:
        return self.y_bounds

    @x_bounds.setter
    def x_bounds(self, x_bounds: BboxBounds) -> None:
        x_lb, x_ub = x_bounds
        assert (
            x_ub > x_lb
        ), f"Expected X upper bound to be strictly greater than lower bounds got bounds: {x_bounds}"
        self._x_bounds = x_bounds

    @y_bounds.setter
    def y_bounds(self, y_bounds: BboxBounds) -> None:
        y_lb, y_ub = y_bounds
        assert (
            y_ub > y_lb
        ), f"Expected Y upper bound to be strictly greater than lower bounds got bounds: {y_bounds}"
        self._y_bounds = y_bounds

    def _get_roi_bounds_in_pixels(
        self, mask: torch.Tensor
    ) -> Tuple[BboxBoundsPixel, BboxBoundsPixel]:
        """
        Gets the bounds for the ROI for an image of a given shape, in pixels.

        Args:
            shape (torch.Tensor): The boolean traversability mask.

        Return:
            Tuple[Bbox_Bounds, Bbox_Bounds]: Bounds of ROI in pixels in the
                format `((x_lb, x_ub), (y_lb, y_ub))` where all bounds are
                `int` values.
        """
        h, w = mask.shape
        x_lb, x_ub = self._x_bounds
        y_lb, y_ub = self._y_bounds
        x_start, x_end = int(w * x_lb), int(w * x_ub) + 1
        y_start, y_end = int(h * y_lb), int(h * y_ub) + 1

        return (x_start, x_end), (y_start, y_end)

    def _get_roi(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Extracts the elements in the boolean traversability mask, which fall
        in the region of interest.

        Args:
            mask (torch.Tensor): The boolean traversability mask.
                Dim: `H x W`, where `H`, `W` are height, width of image.

        Returns:
            torch.Tensor: The boolean traversability mask of the ROI.
        """
        (x_start, x_end), (y_start, y_end) = self._get_roi_bounds_in_pixels(mask=mask)
        return mask[y_start:y_end, x_start:x_end].to(device=self._device)

    def trav_area(self, mask: torch.Tensor) -> float:
        """
        Calculates the fraction of the area in the region of interest that is
        traversable.

        Args:
            mask (torch.Tensor): The traversability mask. Dim: `H x W`, where \
                `H`, `W` are height, width of image.

        Returns:
            float: The mean of the values in the ROI of the mask. If `mask`
                contains `bool` values, this gives the fraction of pixels
                elements in the ROI whose value is `True`. Otherwise, it
                returns the mean of the values of elements in the ROI.

        Note:
            Implementation for the ROI calculation is the same for `bool`
            or otherwise, but the intuition behind the output varies and the
            above description is provided only for exlpanation purposes.
        """
        roi_mask: torch.Tensor = self._get_roi(mask=mask)
        return roi_mask.float().mean().item()
