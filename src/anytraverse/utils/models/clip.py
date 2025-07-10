import torch
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from anytraverse.utils.base.encoding import HuggingFaceImageEncoder
from anytraverse import typing as anyt
from typing import Literal, Sequence
from typing_extensions import override


class CLIPImageEncoder[TImage: anyt.Image](HuggingFaceImageEncoder[TImage]):
    """
    CLIP image encoder wrapper for use with AnyTraverse.
    """

    DIM = 512

    def __init__(
        self,
        model_name: Literal[
            "openai/clip-vit-base-patch32", "openai/clip-vit-base-patch16"
        ],
        cache_dir: str | None = None,
        use_fast: bool = True,
    ) -> None:
        super().__init__(model_name=model_name, cache_dir=cache_dir, use_fast=use_fast)
        self._model = CLIPVisionModelWithProjection.from_pretrained(
            self._model_name, cache_dir=self._cache_dir
        )
        self._processor = AutoProcessor.from_pretrained(
            self._model_name,
            use_fast=self._use_fast_processor,
            cache_dir=self._cache_dir,
        )

    @override
    def __call__(self, x: TImage | Sequence[TImage]) -> torch.Tensor:
        inputs = self._processor(images=x, return_tensors="pt")
        outputs = self._model(**inputs)
        return outputs.image_embeds
