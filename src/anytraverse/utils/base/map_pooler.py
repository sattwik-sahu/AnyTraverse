from abc import ABC, abstractmethod
from typing import Iterable

from anytraverse import typing as anyt


class BaseMapPooler[TInputMap, TPooledMap](ABC):
    """
    Base class for a map pooler.
    """

    @staticmethod
    @abstractmethod
    def pool(maps: Iterable[TInputMap], *args, **kwargs) -> TPooledMap:
        pass


class PromptAttentionMapPooler[TOutputMap](
    BaseMapPooler[anyt.PromptAttentionMap, TOutputMap], ABC
):
    """
    Base pooler class for a map pooler that pools attention maps to a
    given type of map `TOutputMap`.
    """

    @staticmethod
    @abstractmethod
    def pool(
        maps: Iterable[anyt.PromptAttentionMap], prefs: anyt.TraversabilityPreferences
    ) -> TOutputMap:
        pass


type TraversabilityPooler = PromptAttentionMapPooler[anyt.TraversabilityMap]
type UncertaintyPooler = PromptAttentionMapPooler[anyt.UncertaintyMap]
