from abc import ABC, abstractmethod
from typing import Any, Type, TypeVar, Callable

TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


class Model[TInput, TOutput]:
    @abstractmethod
    def __call__(self, x: TInput) -> TOutput:
        pass
