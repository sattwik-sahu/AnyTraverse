from abc import ABC, abstractmethod
from anytraverse import typing as anyt


class BaseAttentionMapping[TInputMap, TPrompt, TOutputMap](ABC):
    """
    The base attention mapping module.

    Produces attention maps of type `TMapOutput` on inputs of type `TMapInput` using
    prompt(s) of type `TPrompt`.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(
        self, x: TInputMap, prompts: TPrompt | list[TPrompt]
    ) -> list[TOutputMap]:
        pass


class PromptAttentionMapping[TImage: anyt.Image](
    BaseAttentionMapping[TImage, anyt.Prompt, anyt.PromptAttentionMap]
):
    pass
