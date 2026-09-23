"""SAM 2 box-to-mask segmenter shared by the detector-based attention mappings."""

import torch
from PIL import Image as PILImage

from anytraverse.models._base import HuggingFaceModel, require_transformers, resize_maps

Box = tuple[float, float, float, float]
"""A box as ``(x1, y1, x2, y2)`` in pixel coordinates."""


class Sam2BoxSegmenter(HuggingFaceModel):
    """
    Turns bounding boxes into soft masks with SAM 2.

    Args:
        model_id (str): SAM 2 checkpoint. Defaults to ``"facebook/sam2.1-hiera-tiny"``.
        device (str | torch.device | None): Device to run on; auto-detected if ``None``.
        dtype (torch.dtype | None): Weight precision; see :class:`HuggingFaceModel`.
        cache_dir (str | None): Override the Hugging Face cache directory.
        compile (bool): Wrap the model with ``torch.compile``.
    """

    def __init__(
        self,
        model_id: str = "facebook/sam2.1-hiera-tiny",
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        cache_dir: str | None = None,
        compile: bool = False,
    ) -> None:
        super().__init__(model_id, device=device, dtype=dtype, cache_dir=cache_dir, compile=compile)
        transformers = require_transformers()
        self._processor = self._load_processor(transformers.Sam2Processor)
        self._model = self._load_model(transformers.Sam2Model)

    @torch.inference_mode()
    def __call__(self, image: PILImage.Image, boxes_per_prompt: list[list[Box]]) -> torch.Tensor:
        """
        Segments the boxes and merges them per prompt.

        The image is encoded once; all boxes are decoded in a single pass.

        Args:
            image (PIL.Image.Image): The RGB image.
            boxes_per_prompt (list[list[Box]]): For each prompt, its detected boxes.

        Returns:
            torch.Tensor: ``(N, H, W)`` maps in ``[0, 1]``, one per prompt. A
                prompt without boxes gets an all-zero map.
        """
        width, height = image.size
        flat_boxes = [box for boxes in boxes_per_prompt for box in boxes]
        maps = torch.zeros(len(boxes_per_prompt), height, width, device=self.device)
        if not flat_boxes:
            return maps

        inputs = self._to_device(
            self._processor(
                images=image, input_boxes=[[list(b) for b in flat_boxes]], return_tensors="pt"
            )
        )
        outputs = self._model(**inputs, multimask_output=False)
        masks = torch.sigmoid(outputs.pred_masks[0, :, 0].float())
        masks = resize_maps(masks, (height, width))

        offset = 0
        for i, boxes in enumerate(boxes_per_prompt):
            if boxes:
                maps[i] = masks[offset : offset + len(boxes)].max(dim=0).values
                offset += len(boxes)
        return maps


__all__ = ["Box", "Sam2BoxSegmenter"]
