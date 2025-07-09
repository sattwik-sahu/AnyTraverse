import torch
from PIL import Image as PILImage
import numpy as np
from numpy import typing as npt
from typing import Callable

type Prompt = str
type Weight = float
type TraversabilityPreferences = dict[Prompt, Weight]
type Image = PILImage.Image | npt.NDArray[np.uint8]
type Encoding = torch.Tensor
type TraversabilityMap = torch.Tensor
type PromptAttentionMap = torch.Tensor
type UncertaintyMap = torch.Tensor
type HistoryElement[TKey] = tuple[TKey, TraversabilityPreferences]
type SimilarityFunction[TElement, TSim] = Callable[[TElement, TElement], TSim]
