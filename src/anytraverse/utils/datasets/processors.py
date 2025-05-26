from anytraverse.utils.datasets import ImageProcessor
from PIL import Image
from typing import List
import numpy as np
from numpy import typing as npt
from anytraverse.config.datasets import RGBValue


class RGBTraversibleMaskProcessor(ImageProcessor):
    """
    Compares each pixel in the image to a set of given rgb values
    for traversible and non-traversible and returns a binary mask
    with `True` for traversible and `False` for non-traversible.

    Attributes:
        _trav_rgb (Tuple[int, int, int]): RGB values for traversible pixels
        _non_trav_rgb (Tuple[int, int, int]): RGB values for non-traversible
            pixels.
    """

    _trav_rgb: List[RGBValue]
    _non_trav_rgb: List[RGBValue]

    def __init__(self, trav_rgb: List[RGBValue], non_trav_rgb: List[RGBValue]) -> None:
        self._trav_rgb = trav_rgb
        self._non_trav_rgb = non_trav_rgb

    def __call__(self, x: Image.Image) -> np.ndarray:
        """
        Compares each pixel in the image to the set of traversible and non-traversible
        RGB values and returns a binary mask with `True` for traversible and `False` for non-traversible.

        Args:
            x (Image.Image): Image to be processed.

        Returns:
            np.ndarray: Binary mask with `True` for traversible and `False` for non-traversible.
        """
        image_array = np.array(x.convert("RGB"))
        return np.sum(
            [np.all(image_array == v, axis=-1) for v in self._trav_rgb],
            axis=0,
            dtype=bool,
        )


class ValueTraversibleMaskProcessor(ImageProcessor):
    """
    Compares each pixel in the image to a set of given values for traversible
    and non-traversible and returns a binary mask with `True` for traversible and
    `False` for non-traversible.

    Attributes:
        _trav_values (List[int]): Values for traversible pixels.
        _non_trav_values (List[int]): Values for non-traversible pixels.
    """

    _trav_values: List[int]
    _non_trav_values: List[int]

    def __init__(self, trav_values: List[int], non_trav_values: List[int]) -> None:
        self._trav_values = trav_values
        self._non_trav_values = non_trav_values

    def __call__(self, x: Image.Image) -> np.ndarray:
        """
        Compares each pixel in the image to the set of traversible and non-traversible
        values and returns a binary mask with `True` for traversible and `False` for non-traversible.

        Args:
            x (Image.Image): Image to be processed.

        Returns:
            np.ndarray: Binary mask with `True` for traversible and `False` for non-traversible.
        """
        image_arr = np.array(x)
        return np.sum([image_arr == v for v in self._trav_values], axis=0) > 0
