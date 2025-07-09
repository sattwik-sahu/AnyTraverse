import torch
from numpy import typing as npt
from typing import TypeVar


TMat = TypeVar("TMat", torch.Tensor, npt.NDArray)


class RegionOfInterest:
    """
    Class to handle operations on the region of interest (ROI)
    of a given matrix
    """

    def __init__(
        self, x_bounds: tuple[float, float], y_bounds: tuple[float, float]
    ) -> None:
        self._x_bounds = x_bounds
        self._y_bounds = y_bounds

    def extract(self, mat: TMat) -> tuple[TMat, tuple[int, int], tuple[int, int]]:
        """
        Extracts the region of interest and the pixel coordinates of the ROI as a box.

        Args:
            mat (torch.Tensor | npt.NDArray): The image of map to find the ROI of as a
        """
        height, width = mat.shape[-2:]
        x_start, x_end = [int(width * bound) for bound in self._x_bounds]
        y_start, y_end = [int(height * bound) for bound in self._y_bounds]
        return (
            mat[y_start : y_end + 1, x_start : x_end + 1],
            (x_start, y_start),
            (x_end, y_end),
        )
