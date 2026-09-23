"""SAM 3 prompt attention mapping."""

from collections.abc import Sequence
from typing import Any, Literal, override

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


class SAM3AttentionMapping(HuggingFaceModel, PromptAttentionMapping):
    """
    Prompt attention maps from SAM 3 (Segment Anything with Concepts).

    SAM 3 segments every instance of a short noun phrase. The image is encoded
    once per frame and the prompt embeddings are cached, so each extra prompt
    only costs a pass through the detector and mask decoder.

    The ``facebook/sam3`` weights are gated: accept the license on the Hugging
    Face Hub and run ``hf auth login`` once before first use.

    Args:
        model_id (str): SAM 3 checkpoint. Defaults to ``"facebook/sam3"``.
        mode (Literal["semantic", "instance"]): ``"semantic"`` uses the model's
            single-channel semantic head. ``"instance"`` takes the per-pixel
            maximum over instance masks whose score exceeds ``instance_threshold``.
        instance_threshold (float): Minimum instance score in ``"instance"`` mode.
        image_size (int | None): Square input resolution. The model is trained
            at 1008; 560 roughly halves memory and latency at some accuracy cost.
        device (str | torch.device | None): Device to run on; auto-detected if ``None``.
        dtype (torch.dtype | None): Weight precision; see :class:`HuggingFaceModel`.
        cache_dir (str | None): Override the Hugging Face cache directory.
        compile (bool): Wrap the model with ``torch.compile``.
    """

    def __init__(
        self,
        model_id: str = "facebook/sam3",
        mode: Literal["semantic", "instance"] = "semantic",
        instance_threshold: float = 0.5,
        image_size: int | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        cache_dir: str | None = None,
        compile: bool = False,
    ) -> None:
        super().__init__(model_id, device=device, dtype=dtype, cache_dir=cache_dir, compile=compile)
        if mode not in ("semantic", "instance"):
            raise ValueError(f"mode must be 'semantic' or 'instance', got {mode!r}")
        self.mode = mode
        self.instance_threshold = instance_threshold
        transformers = require_transformers()

        model_kwargs: dict[str, Any] = {}
        processor_kwargs: dict[str, Any] = {}
        if image_size is not None:
            config = transformers.Sam3Config.from_pretrained(model_id, cache_dir=cache_dir)
            config.image_size = image_size
            model_kwargs["config"] = config
            processor_kwargs["size"] = {"height": image_size, "width": image_size}
        self._processor = self._load_processor(transformers.Sam3Processor, **processor_kwargs)
        self._model = self._load_model(transformers.Sam3Model, **model_kwargs)
        self._prompt_cache: dict[str, tuple[Any, torch.Tensor]] = {}

    def _text_features(self, prompt: str) -> tuple[Any, torch.Tensor]:
        """Returns the cached ``(text_embeds, attention_mask)`` for a prompt."""
        if prompt not in self._prompt_cache:
            if len(self._prompt_cache) >= 64:
                self._prompt_cache.clear()
            text_inputs = self._processor(text=prompt, return_tensors="pt").to(self.device)
            text_embeds = self._model.get_text_features(**text_inputs)
            self._prompt_cache[prompt] = (text_embeds, text_inputs["attention_mask"])
        return self._prompt_cache[prompt]

    def _map_from_outputs(self, outputs: Any) -> torch.Tensor:
        """Reduces one forward pass to a single ``(h, w)`` map in ``[0, 1]``."""
        if self.mode == "semantic":
            return torch.sigmoid(outputs.semantic_seg[0, 0].float())
        scores = torch.sigmoid(outputs.pred_logits[0].float())
        masks = torch.sigmoid(outputs.pred_masks[0].float())
        keep = scores > self.instance_threshold
        if not keep.any():
            return torch.zeros_like(masks[0])
        return masks[keep].max(dim=0).values

    @override
    @torch.inference_mode()
    def __call__(
        self, x: anyt.Image, prompts: anyt.Prompt | Sequence[anyt.Prompt]
    ) -> list[anyt.PromptAttentionMap]:
        image = to_pil(x)
        prompts = as_prompt_list(prompts)
        width, height = image.size

        pixel_values = self._to_device(self._processor(images=image, return_tensors="pt"))[
            "pixel_values"
        ]
        vision_embeds = self._model.get_vision_features(pixel_values=pixel_values)

        maps = []
        for prompt in prompts:
            text_embeds, attention_mask = self._text_features(prompt)
            outputs = self._model(
                vision_embeds=vision_embeds,
                text_embeds=text_embeds,
                attention_mask=attention_mask,
            )
            maps.append(self._map_from_outputs(outputs))
        return list(resize_maps(torch.stack(maps), (height, width)).unbind(0))


__all__ = ["SAM3AttentionMapping"]
