from abc import ABC, abstractmethod

import torch
from typing_extensions import override

from anytraverse import typing as anyt


class BaseHistory[TKey](ABC):
    """
    The base class to manage the history.
    """

    def __init__(self) -> None:
        self._store: list[anyt.HistoryElement[TKey]] = []

    @property
    def store(self) -> list[anyt.HistoryElement[TKey]]:
        return self._store

    @abstractmethod
    def find_best_match(
        self, query: TKey, *args, **kwargs
    ) -> anyt.HistoryElement[TKey]:
        pass

    def add(
        self, key: TKey, traversabilty_preferences: anyt.TraversabilityPreferences
    ) -> anyt.HistoryElement[TKey]:
        element: anyt.HistoryElement = (key, traversabilty_preferences)
        self._store.append(element)
        return element


class SimilarityHistory[TKey, TSim](BaseHistory[TKey], ABC):
    """
    A history that finds the best match based on a similarity function.
    """

    def __init__(self, similarity_func: anyt.SimilarityFunction[TKey, TSim]) -> None:
        super().__init__()
        self._similarity_func = similarity_func


class EncodingHistory(SimilarityHistory[anyt.Encoding, torch.Tensor]):
    """
    Stores the history as a pairs of encodings and traversability preferences.
    """

    def __init__(
        self,
        similarity_func: anyt.SimilarityFunction[anyt.Encoding, torch.Tensor],
    ) -> None:
        super().__init__(similarity_func=similarity_func)

    def _get_encodings(self) -> torch.Tensor:
        return torch.cat([encoding.view(1, -1) for (encoding, _) in self._store], dim=0)

    @override
    def find_best_match(
        self, query: anyt.Encoding
    ) -> anyt.HistoryElement[anyt.Encoding]:
        similarities = self._similarity_func(query, self._get_encodings()).ravel()
        best_match_inx = int(similarities.argmax())
        best_match = self._store[best_match_inx]
        return best_match
