"""Type aliases shared across the AnyTraverse package."""

from collections.abc import Callable

import numpy as np
import torch
from numpy import typing as npt
from PIL import Image as PILImage

type Prompt = str
"""A natural-language prompt describing a terrain class, e.g. ``"road"``."""

type Weight = float
"""Traversability weight for a prompt in the range ``[-1, 1]``."""

type TraversabilityPreferences = dict[Prompt, Weight]
"""Mapping from prompts to weights, e.g. ``{"road": 1.0, "bush": -0.8}``."""

type Image = PILImage.Image | npt.NDArray[np.uint8]
"""An RGB image, either a PIL image or a ``uint8`` array of shape ``(H, W, 3)``."""

type Encoding = torch.Tensor
"""A dense image embedding of shape ``(1, D)``."""

type PromptAttentionMap = torch.Tensor
"""Per-pixel score in ``[0, 1]`` for one prompt, shape ``(H, W)``."""

type TraversabilityMap = torch.Tensor
"""Per-pixel traversability in ``[0, 1]``, shape ``(H, W)``."""

type UncertaintyMap = torch.Tensor
"""Per-pixel uncertainty in ``[0, 1]``, shape ``(H, W)``."""

type HistoryElement[TKey] = tuple[TKey, TraversabilityPreferences]
"""A stored ``(key, preferences)`` pair in the scene history."""

type SimilarityFunction[TElement, TSim] = Callable[[TElement, TElement], TSim]
"""A function returning the similarity between two elements."""

__all__ = [
    "Encoding",
    "HistoryElement",
    "Image",
    "Prompt",
    "PromptAttentionMap",
    "SimilarityFunction",
    "TraversabilityMap",
    "TraversabilityPreferences",
    "UncertaintyMap",
    "Weight",
]
