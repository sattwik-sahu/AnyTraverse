import torch
from PIL import Image as PILImage
from torchvision.transforms import Resize
from transformers import AutoProcessor, CLIPSegForImageSegmentation
from typing_extensions import override

from anytraverse.utils import _typing as anyt
from anytraverse.helpers.device import DEVICE
from anytraverse.utils.base.attention_mapping import PromptAttentionMapping

from anytraverse.helpers.os_ import PLATFORM, PlatformType


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
        # Convert the image to PIL Image, if not already
        if not isinstance(x, PILImage.Image):
            x = PILImage.fromarray(x)

        # Prompts must be in a list format
        if isinstance(prompts, str):
            prompts = [prompts]

        # Resize transform to bring output maps to same size as input image
        width, height = x.size
        resize = Resize(size=(height, width)).to(device=self._device)

        with torch.inference_mode():
            if (PLATFORM is PlatformType.LINUX) or (PLATFORM is PlatformType.WINDOWS):
                # Normal batched inference on Windows and Linux
                inputs = self._processor(
                    text=prompts,
                    images=[x] * len(prompts),
                    padding=True,
                    return_tensors="pt",
                ).to(device=self._device)
                # Get the prompt attention maps one by one, apply sigmoid
                maps = [m for m in torch.sigmoid(resize(self._model(**inputs).logits))]
            elif PLATFORM is PlatformType.MAC:
                # Create the empty prompt attention maps list
                maps = []
                for prompt in prompts:
                    # Preprocess inputs, one prompt at a time for Mac
                    inputs = self._processor(
                        text=prompt,
                        images=x,
                        padding=True,
                        return_tensors="pt",
                    ).to(device=self._device)
                    # Perform inference
                    output = torch.sigmoid(
                        resize(self._model(**inputs).logits)
                    ).squeeze(0)
                    # Add the prompt attention map to the list
                    maps.append(output)
            else:
                # Unknown platform
                maps = []

        return maps
