from abc import ABC, abstractmethod

from anytraverse import typing as anyt


class BaseMapPooler[TInputMap, TPooledMap](ABC):
    """
    Base class for a map pooler.
    """

    @staticmethod
    @abstractmethod
    def pool(maps: list[TInputMap], *args, **kwargs) -> TPooledMap:
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
        maps: list[anyt.PromptAttentionMap],
        traversability_preferences: anyt.TraversabilityPreferences,
    ) -> TOutputMap:
        pass


TraversabilityPooler = PromptAttentionMapPooler[anyt.TraversabilityMap]
UncertaintyPooler = PromptAttentionMapPooler[anyt.UncertaintyMap]
