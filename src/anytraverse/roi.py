"""Region of interest extraction from maps."""

from typing import TypeVar

import torch
from numpy import typing as npt

TMat = TypeVar("TMat", torch.Tensor, npt.NDArray)


class RegionOfInterest:
    """
    A rectangular region of interest expressed as fractions of the image size.

    Args:
        x_bounds (tuple[float, float]): ``(start, end)`` fractions of the width.
        y_bounds (tuple[float, float]): ``(start, end)`` fractions of the height.

    Example:
        The bottom-centre patch in front of a robot::

            roi = RegionOfInterest(x_bounds=(0.333, 0.667), y_bounds=(0.6, 0.95))
    """

    def __init__(self, x_bounds: tuple[float, float], y_bounds: tuple[float, float]) -> None:
        for name, (lo, hi) in (("x_bounds", x_bounds), ("y_bounds", y_bounds)):
            if not (0.0 <= lo < hi <= 1.0):
                raise ValueError(f"{name} must satisfy 0 <= start < end <= 1, got {(lo, hi)}")
        self._x_bounds = x_bounds
        self._y_bounds = y_bounds

    @property
    def x_bounds(self) -> tuple[float, float]:
        """Fractional ``(start, end)`` bounds along the width."""
        return self._x_bounds

    @property
    def y_bounds(self) -> tuple[float, float]:
        """Fractional ``(start, end)`` bounds along the height."""
        return self._y_bounds

    def pixel_bounds(self, height: int, width: int) -> tuple[tuple[int, int], tuple[int, int]]:
        """
        Converts the fractional bounds to inclusive pixel coordinates.

        Args:
            height (int): Image height in pixels.
            width (int): Image width in pixels.

        Returns:
            tuple[tuple[int, int], tuple[int, int]]: ``((x_start, y_start), (x_end, y_end))``.
        """
        x_start, x_end = (int(width * b) for b in self._x_bounds)
        y_start, y_end = (int(height * b) for b in self._y_bounds)
        x_end = min(x_end, width - 1)
        y_end = min(y_end, height - 1)
        return (x_start, y_start), (x_end, y_end)

    def extract(self, mat: TMat) -> tuple[TMat, tuple[int, int], tuple[int, int]]:
        """
        Crops the region of interest from a map.

        Args:
            mat (torch.Tensor | np.ndarray): A map of shape ``(H, W)``.

        Returns:
            tuple: ``(crop, (x_start, y_start), (x_end, y_end))`` where the end
                coordinates are inclusive.
        """
        height, width = mat.shape[:2]
        (x_start, y_start), (x_end, y_end) = self.pixel_bounds(height, width)
        return mat[y_start : y_end + 1, x_start : x_end + 1], (x_start, y_start), (x_end, y_end)


__all__ = ["RegionOfInterest"]
