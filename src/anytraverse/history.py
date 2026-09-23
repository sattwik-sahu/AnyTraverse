"""Scene history backed by image encodings."""

from typing import override

import torch

from anytraverse import typing as anyt
from anytraverse.interfaces import History


class EncodingHistory(History[anyt.Encoding]):
    """
    Stores image encodings with their traversability preferences.

    Encodings are kept stacked in a single ``(N, D)`` tensor so that a lookup
    is one similarity call instead of ``N``.

    Args:
        similarity_func (SimilarityFunction[Encoding, torch.Tensor]): Called as
            ``similarity_func(query, encodings)`` with ``query`` of shape
            ``(1, D)`` and ``encodings`` of shape ``(N, D)``; must return a
            tensor with ``N`` similarity values. ``torch.cosine_similarity``
            satisfies this.
        max_size (int | None): Keep only the most recent ``max_size`` scenes.
    """

    def __init__(
        self,
        similarity_func: anyt.SimilarityFunction[anyt.Encoding, torch.Tensor],
        max_size: int | None = None,
    ) -> None:
        super().__init__(max_size=max_size)
        self._similarity_func = similarity_func
        self._encodings: torch.Tensor | None = None

    @property
    def encodings(self) -> torch.Tensor:
        """All stored encodings stacked into a ``(N, D)`` tensor."""
        if self._encodings is None:
            raise ValueError("History is empty")
        return self._encodings

    @override
    def add(
        self, key: anyt.Encoding, traversability_preferences: anyt.TraversabilityPreferences
    ) -> anyt.HistoryElement[anyt.Encoding]:
        key = key.detach().reshape(1, -1)
        element = super().add(key, traversability_preferences)
        if self._encodings is None:
            self._encodings = key
        else:
            self._encodings = torch.cat([self._encodings, key.to(self._encodings)], dim=0)
        return element

    @override
    def _evict_oldest(self) -> None:
        super()._evict_oldest()
        assert self._encodings is not None  # eviction implies a non-empty history
        self._encodings = self._encodings[1:]

    @override
    def find_best_match(
        self, query: anyt.Encoding
    ) -> tuple[anyt.HistoryElement[anyt.Encoding], float]:
        if self._encodings is None:
            raise ValueError("Cannot search an empty history")
        query = query.reshape(1, -1).to(self._encodings)
        similarities = self._similarity_func(query, self._encodings).ravel()
        best_index = int(similarities.argmax())
        return self._store[best_index], float(similarities[best_index])


__all__ = ["EncodingHistory"]
