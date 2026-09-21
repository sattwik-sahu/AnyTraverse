"""
Abstract interfaces for the swappable parts of the AnyTraverse pipeline.

Implement these to bring your own vision-language model, image encoder or
pooling strategy. The pipeline only depends on the contracts defined here.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch

from anytraverse import typing as anyt


class PromptAttentionMapping(ABC):
    """
    Produces one attention map per prompt for an image.

    An attention map is a ``torch.Tensor`` of shape ``(H, W)`` with values in
    ``[0, 1]`` giving how strongly each pixel matches the prompt. ``H`` and
    ``W`` must equal the height and width of the input image.
    """

    @abstractmethod
    def __call__(
        self, x: anyt.Image, prompts: anyt.Prompt | Sequence[anyt.Prompt]
    ) -> list[anyt.PromptAttentionMap]:
        """
        Computes the attention maps.

        Args:
            x (Image): The input RGB image.
            prompts (Prompt | Sequence[Prompt]): One prompt or a list of prompts.

        Returns:
            list[PromptAttentionMap]: One ``(H, W)`` map per prompt, in the
                same order as the prompts.
        """


class ImageEncoder(ABC):
    """
    Encodes an image into a dense vector used for scene similarity.

    Attributes:
        dim (int): Dimensionality of the produced encoding.
    """

    dim: int

    @abstractmethod
    def __call__(self, x: anyt.Image | Sequence[anyt.Image]) -> anyt.Encoding:
        """
        Encodes one or more images.

        Args:
            x (Image | Sequence[Image]): A single image or a batch of images.

        Returns:
            Encoding: A tensor of shape ``(B, dim)`` where ``B`` is the number
                of images (``1`` for a single image).
        """


class PromptAttentionMapPooler[TOutputMap: torch.Tensor](ABC):
    """
    Pools per-prompt attention maps into a single map.

    Subclasses implement :meth:`pool` as a static method so they can be passed
    to the pipeline as classes without instantiation.
    """

    @staticmethod
    @abstractmethod
    def pool(
        maps: list[anyt.PromptAttentionMap],
        traversability_preferences: anyt.TraversabilityPreferences,
    ) -> TOutputMap:
        """
        Pools the attention maps.

        Args:
            maps (list[PromptAttentionMap]): One ``(H, W)`` map per prompt, in
                the order of ``traversability_preferences``.
            traversability_preferences (TraversabilityPreferences): The current
                prompts and their weights.

        Returns:
            The pooled ``(H, W)`` map.
        """


TraversabilityPooler = PromptAttentionMapPooler[anyt.TraversabilityMap]
"""Pooler producing a traversability map."""

UncertaintyPooler = PromptAttentionMapPooler[anyt.UncertaintyMap]
"""Pooler producing an uncertainty map."""


class History[TKey](ABC):
    """
    Stores ``(key, traversability preferences)`` pairs and retrieves the best match.

    Args:
        max_size (int | None): If given, only the most recent ``max_size``
            elements are kept.
    """

    def __init__(self, max_size: int | None = None) -> None:
        if max_size is not None and max_size < 1:
            raise ValueError("max_size must be a positive integer or None")
        self._max_size = max_size
        self._store: list[anyt.HistoryElement[TKey]] = []

    @property
    def store(self) -> list[anyt.HistoryElement[TKey]]:
        """The stored elements, oldest first."""
        return self._store

    def __len__(self) -> int:
        return len(self._store)

    def add(
        self, key: TKey, traversability_preferences: anyt.TraversabilityPreferences
    ) -> anyt.HistoryElement[TKey]:
        """
        Appends an element to the history.

        Args:
            key (TKey): The key used for matching, e.g. an image encoding.
            traversability_preferences (TraversabilityPreferences): The
                preferences to associate with the key. A copy is stored.

        Returns:
            HistoryElement[TKey]: The stored element.
        """
        element: anyt.HistoryElement[TKey] = (key, dict(traversability_preferences))
        self._store.append(element)
        if self._max_size is not None and len(self._store) > self._max_size:
            self._evict_oldest()
        return element

    def _evict_oldest(self) -> None:
        """Removes the oldest element. Subclasses may extend this."""
        del self._store[0]

    @abstractmethod
    def find_best_match(self, query: TKey) -> tuple[anyt.HistoryElement[TKey], float]:
        """
        Finds the stored element most similar to ``query``.

        Args:
            query (TKey): The query key.

        Returns:
            tuple[HistoryElement[TKey], float]: The best element and its
                similarity to the query.

        Raises:
            ValueError: If the history is empty.
        """


__all__ = [
    "History",
    "ImageEncoder",
    "PromptAttentionMapPooler",
    "PromptAttentionMapping",
    "TraversabilityPooler",
    "UncertaintyPooler",
]
