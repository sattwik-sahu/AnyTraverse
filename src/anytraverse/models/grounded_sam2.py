"""Grounding DINO + SAM 2 prompt attention mapping."""

from collections.abc import Sequence
from typing import override

import torch

from anytraverse import typing as anyt
from anytraverse.interfaces import PromptAttentionMapping
from anytraverse.models._base import (
    HuggingFaceModel,
    as_prompt_list,
    require_transformers,
    to_pil,
)
from anytraverse.models._sam2 import Box, Sam2BoxSegmenter


class GroundedSAM2AttentionMapping(HuggingFaceModel, PromptAttentionMapping):
    """
    Prompt attention maps from Grounding DINO detections refined by SAM 2.

    All prompts are detected in a single Grounding DINO pass by joining them
    with ``"."``; each detection is assigned to the prompt whose tokens it
    scores highest on. The boxes are then turned into soft masks by SAM 2 and
    merged per prompt with a per-pixel maximum. Prompts with no detections
    get an all-zero map.

    Args:
        model_id (str): Grounding DINO checkpoint. Defaults to
            ``"IDEA-Research/grounding-dino-tiny"``.
        sam2_model_id (str): SAM 2 checkpoint. Defaults to ``"facebook/sam2.1-hiera-tiny"``.
        box_threshold (float): Minimum detection score to keep a box.
        device (str | torch.device | None): Device to run on; auto-detected if ``None``.
        dtype (torch.dtype | None): Weight precision; see :class:`HuggingFaceModel`.
        cache_dir (str | None): Override the Hugging Face cache directory.
        compile (bool): Wrap both models with ``torch.compile``.
    """

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        sam2_model_id: str = "facebook/sam2.1-hiera-tiny",
        box_threshold: float = 0.3,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        cache_dir: str | None = None,
        compile: bool = False,
    ) -> None:
        super().__init__(model_id, device=device, dtype=dtype, cache_dir=cache_dir, compile=compile)
        transformers = require_transformers()
        self.box_threshold = box_threshold
        self._processor = self._load_processor(transformers.GroundingDinoProcessor)
        self._model = self._load_model(transformers.GroundingDinoForObjectDetection)
        self._segmenter = Sam2BoxSegmenter(
            sam2_model_id,
            device=self.device,
            dtype=self.dtype,
            cache_dir=cache_dir,
            compile=compile,
        )
        tokenizer = self._processor.tokenizer
        self._separator_id = int(tokenizer.convert_tokens_to_ids("."))
        self._special_ids = set(tokenizer.all_special_ids)

    def _prompt_token_masks(
        self, input_ids: torch.Tensor, num_prompts: int, length: int | None = None
    ) -> torch.Tensor:
        """
        Returns a ``(num_prompts, L)`` boolean mask marking which tokens belong to which prompt.

        Prompts are separated by the ``"."`` token; special tokens are ignored.
        The model pads text features to a fixed width (256), so ``length`` sets the
        mask width and any positions past the input are left ``False``.
        """
        length = input_ids.numel() if length is None else length
        masks = torch.zeros(num_prompts, length, dtype=torch.bool, device=self.device)
        prompt_index = 0
        for position, token_id in enumerate(input_ids.tolist()):
            if token_id in self._special_ids:
                continue
            if token_id == self._separator_id:
                prompt_index += 1
                continue
            if prompt_index < num_prompts:
                masks[prompt_index, position] = True
        return masks

    def _detect(self, image, prompts: list[str]) -> list[list[Box]]:
        """Runs Grounding DINO once and groups the boxes by prompt."""
        text = ". ".join(p.strip().rstrip(".") for p in prompts) + "."
        inputs = self._to_device(self._processor(images=image, text=text, return_tensors="pt"))
        outputs = self._model(**inputs)

        probs = torch.sigmoid(outputs.logits[0].float())  # (Q, L)
        token_masks = self._prompt_token_masks(
            inputs["input_ids"][0], len(prompts), length=probs.shape[-1]
        )
        # (Q, N): best token score of each query for each prompt.
        scores = torch.stack(
            [
                probs[:, mask].max(dim=-1).values if mask.any() else probs.new_zeros(probs.shape[0])
                for mask in token_masks
            ],
            dim=-1,
        )
        best_scores, best_prompts = scores.max(dim=-1)
        keep = best_scores > self.box_threshold

        width, height = image.size
        boxes = outputs.pred_boxes[0].float()[keep]  # (K, 4) as normalized cx, cy, w, h
        cx, cy, w, h = boxes.unbind(-1)
        corners = torch.stack(
            [
                (cx - w / 2) * width,
                (cy - h / 2) * height,
                (cx + w / 2) * width,
                (cy + h / 2) * height,
            ],
            dim=-1,
        )

        boxes_per_prompt: list[list[Box]] = [[] for _ in prompts]
        for box, prompt_index in zip(corners.tolist(), best_prompts[keep].tolist(), strict=True):
            boxes_per_prompt[prompt_index].append(tuple(box))
        return boxes_per_prompt

    @override
    @torch.inference_mode()
    def __call__(
        self, x: anyt.Image, prompts: anyt.Prompt | Sequence[anyt.Prompt]
    ) -> list[anyt.PromptAttentionMap]:
        image = to_pil(x)
        prompts = as_prompt_list(prompts)
        boxes_per_prompt = self._detect(image, prompts)
        return list(self._segmenter(image, boxes_per_prompt).unbind(0))


__all__ = ["GroundedSAM2AttentionMapping"]
