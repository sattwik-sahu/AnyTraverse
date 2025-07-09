import torch
from PIL import Image as PILImage
from torchvision.transforms import Resize
from transformers import AutoProcessor, CLIPSegForImageSegmentation
from typing_extensions import override

from anytraverse import typing as anyt
from anytraverse.helpers.device import DEVICE
from anytraverse.utils.base.attention_mapping import PromptAttentionMapping


class CLIPSegAttentionMapping[TImage: anyt.Image](PromptAttentionMapping[TImage]):
    def __init__(
        self,
        device: torch.device = DEVICE,
        model_name: str = "mcmonkey/clipseg-rd64-refined-fp16",
    ) -> None:
        super().__init__()
        self._device = device
        self._processor = AutoProcessor.from_pretrained(
            model_name, use_fast=True, cache_dir="data/weights/clipseg"
        )
        self._model = CLIPSegForImageSegmentation.from_pretrained(
            model_name, cache_dir="data/weights/clipseg", device_map=str(device)
        )

    @override
    def __call__(
        self, x: TImage, prompts: str | list[str]
    ) -> list[anyt.PromptAttentionMap]:
        if not isinstance(x, PILImage.Image):
            x = PILImage.fromarray(x)
        if isinstance(prompts, str):
            prompts = [prompts]

        width, height = x.size
        resize = Resize(size=(height, width)).to(device=self._device)

        with torch.no_grad():
            inputs = self._processor(
                text=prompts,
                images=[x] * len(prompts),
                padding=True,
                return_tensors="pt",
            ).to(device=self._device)
            maps = [m for m in torch.sigmoid(resize(self._model(**inputs).logits))]
            return maps
