"""CLIPSeg prompt attention mapping."""

from collections.abc import Sequence
from typing import override

import torch

from anytraverse import typing as anyt
from anytraverse.interfaces import PromptAttentionMapping
from anytraverse.models._base import (
    HuggingFaceModel,
    as_prompt_list,
    require_transformers,
    resize_maps,
    to_pil,
)


class CLIPSegAttentionMapping(HuggingFaceModel, PromptAttentionMapping):
    """
    Prompt attention maps from CLIPSeg, the VLM used in the paper.

    The image is passed through the CLIP vision tower once per frame and the
    prompt embeddings are cached, so adding prompts costs only a pass through
    the lightweight decoder.

    Args:
        model_id (str): CLIPSeg checkpoint. Defaults to ``"CIDAS/clipseg-rd64-refined"``.
        device (str | torch.device | None): Device to run on; auto-detected if ``None``.
        dtype (torch.dtype | None): Weight precision; see :class:`HuggingFaceModel`.
        cache_dir (str | None): Override the Hugging Face cache directory.
        compile (bool): Wrap the model with ``torch.compile``.
    """

    def __init__(
        self,
        model_id: str = "CIDAS/clipseg-rd64-refined",
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        cache_dir: str | None = None,
        compile: bool = False,
    ) -> None:
        super().__init__(model_id, device=device, dtype=dtype, cache_dir=cache_dir, compile=compile)
        transformers = require_transformers()
        self._processor = self._load_processor(transformers.AutoProcessor)
        self._model = self._load_model(transformers.CLIPSegForImageSegmentation)
        self._prompt_cache: dict[tuple[str, ...], torch.Tensor] = {}

    def _conditional_embeddings(self, prompts: list[str]) -> torch.Tensor:
        """Returns the ``(N, D)`` CLIP text embeddings for the prompts, cached by prompt tuple."""
        key = tuple(prompts)
        if key not in self._prompt_cache:
            if len(self._prompt_cache) >= 64:
                self._prompt_cache.clear()
            text_inputs = self._processor.tokenizer(prompts, padding=True, return_tensors="pt")
            text_inputs = text_inputs.to(self.device)
            self._prompt_cache[key] = self._model.get_conditional_embeddings(
                batch_size=len(prompts),
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
            )
        return self._prompt_cache[key]

    @override
    @torch.inference_mode()
    def __call__(
        self, x: anyt.Image, prompts: anyt.Prompt | Sequence[anyt.Prompt]
    ) -> list[anyt.PromptAttentionMap]:
        image = to_pil(x)
        prompts = as_prompt_list(prompts)
        width, height = image.size

        pixel_values = self._to_device(
            self._processor.image_processor(images=image, return_tensors="pt")
        )["pixel_values"]
        vision_outputs = self._model.clip.vision_model(pixel_values, output_hidden_states=True)
        hidden_states = vision_outputs.hidden_states
        activations = [
            hidden_states[i + 1].expand(len(prompts), -1, -1) for i in self._model.extract_layers
        ]
        conditional = self._conditional_embeddings(prompts).to(activations[0].dtype)
        logits = self._model.decoder(activations, conditional).logits
        maps = resize_maps(torch.sigmoid(logits.float()), (height, width))
        return list(maps.unbind(0))


__all__ = ["CLIPSegAttentionMapping"]
