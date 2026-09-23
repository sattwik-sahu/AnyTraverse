"""OWLv2 + SAM 2 prompt attention mapping."""

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


class OWLv2SAM2AttentionMapping(HuggingFaceModel, PromptAttentionMapping):
    """
    Prompt attention maps from OWLv2 detections refined by SAM 2.

    OWLv2 scores every prompt against every candidate box in one pass, which
    makes it a faster alternative to Grounding DINO. Boxes above
    ``box_threshold`` are turned into soft masks by SAM 2 and merged per
    prompt with a per-pixel maximum. Prompts with no detections get an
    all-zero map.

    Args:
        model_id (str): OWLv2 checkpoint. Defaults to
            ``"google/owlv2-base-patch16-ensemble"``.
        sam2_model_id (str): SAM 2 checkpoint. Defaults to ``"facebook/sam2.1-hiera-tiny"``.
        box_threshold (float): Minimum detection score to keep a box.
        device (str | torch.device | None): Device to run on; auto-detected if ``None``.
        dtype (torch.dtype | None): Weight precision; see :class:`HuggingFaceModel`.
        cache_dir (str | None): Override the Hugging Face cache directory.
        compile (bool): Wrap both models with ``torch.compile``.
    """

    def __init__(
        self,
        model_id: str = "google/owlv2-base-patch16-ensemble",
        sam2_model_id: str = "facebook/sam2.1-hiera-tiny",
        box_threshold: float = 0.2,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        cache_dir: str | None = None,
        compile: bool = False,
    ) -> None:
        super().__init__(model_id, device=device, dtype=dtype, cache_dir=cache_dir, compile=compile)
        transformers = require_transformers()
        self.box_threshold = box_threshold
        self._processor = self._load_processor(transformers.Owlv2Processor)
        self._model = self._load_model(transformers.Owlv2ForObjectDetection)
        self._segmenter = Sam2BoxSegmenter(
            sam2_model_id,
            device=self.device,
            dtype=self.dtype,
            cache_dir=cache_dir,
            compile=compile,
        )

    def _detect(self, image, prompts: list[str]) -> list[list[Box]]:
        """Runs OWLv2 once and groups the boxes by prompt."""
        inputs = self._to_device(self._processor(images=image, text=[prompts], return_tensors="pt"))
        outputs = self._model(**inputs)
        # OWLv2 pads the image to a square before resizing, so boxes are relative to the
        # padded square whose side is the longer image side.
        width, height = image.size
        side = max(width, height)
        results = self._processor.post_process_grounded_object_detection(
            outputs, threshold=self.box_threshold, target_sizes=[(side, side)]
        )[0]

        boxes_per_prompt: list[list[Box]] = [[] for _ in prompts]
        for box, label in zip(results["boxes"].tolist(), results["labels"].tolist(), strict=True):
            x1, y1, x2, y2 = box
            boxes_per_prompt[label].append(
                (max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2))
            )
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


__all__ = ["OWLv2SAM2AttentionMapping"]
