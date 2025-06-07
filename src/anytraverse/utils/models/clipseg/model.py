from transformers import CLIPSegProcessor, PreTrainedModel, BatchEncoding
import torch
from anytraverse.utils.models.clipseg.loader import load_clipseg_processor_and_model
from typing import Tuple, Dict, Any, List, Literal
from PIL.Image import Image, open as open_image, Resampling
from pathlib import Path
from anytraverse.utils.helpers import DEVICE


torch.set_default_device(device=DEVICE)


CLIPSegInputEncoded = BatchEncoding | Dict[str, Any]


class CLIPSeg:
    """
    Inference wrapper for CLIPSeg model on HuggingFace.
    """

    _processor: CLIPSegProcessor | Tuple[CLIPSegProcessor, Dict[str, Any]]
    _model: PreTrainedModel
    _device: torch.device

    def __init__(
        self,
        model_name: str | None = None,
        device: Literal["cpu", "cuda", "mps"] | torch.device = "cpu",
    ) -> None:
        self._device = torch.device(device)
        self._processor, self._model = load_clipseg_processor_and_model(
            pretrained_model_name=model_name
        )
        self._model = self._model.to(device=DEVICE)  # type: ignore

    def _preprocess(
        self, prompts: List[str], image: Image | Path | str
    ) -> Tuple[Image, CLIPSegInputEncoded]:
        if isinstance(image, Path) or isinstance(image, str):
            image = open_image(image)
        image_resized = image.resize(size=(224, 224), resample=Resampling.LANCZOS)
        images: List[Image] = [image_resized] * len(prompts)
        x = self._processor(
            text=prompts, images=images, return_tensors="pt", padding=True
        ).to(  # type: ignore
            device=self._device
        )
        x["pixel_values"] = torch.nn.functional.interpolate(
            x["pixel_values"],
            size=(224, 224),
            mode="bicubic",
            align_corners=True,
        )
        return image, x.to(device=DEVICE)

    def _run(self, x: CLIPSegInputEncoded) -> torch.Tensor:
        # No gradients
        with torch.no_grad():
            preds = self._model(**x)
        return preds.logits.unsqueeze(1)

    def _postprocess(
        self, y: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        y_proba = torch.sigmoid(y)
        y_proba_resized = torch.nn.functional.interpolate(
            y_proba,
            size=image_size,
            mode="bicubic",
            align_corners=True,
        )
        return y_proba_resized

    def __call__(self, prompts: List[str], image: Image | Path | str) -> torch.Tensor:
        """
        Runs the CLIPSeg model on the given list of prompts and the image.
        The dimensions of the output are `(N, 1, H, W)`, where:
            - `N` is the number of prompts.
            - `H` is the height of the input image.
            - `W` is the width of the input image.

        Args:
            prompts (List[str]): The list of prompt strings.
            image (Image | Path | str): Either the PIL.Image.Image object or
                the Path object pointing to the image file or the string path
                to the image file.

        Returns:
            torch.Tensor:
                The model output, with dimensions `(N, 1, H, W)`
        """
        masks: List[str] = []
        for prompt in prompts:
            image, x = self._preprocess(prompts=[prompt], image=image)
            y = self._run(x=x)
            image_size = image.size[:-3:-1]
            output = self._postprocess(y=y, image_size=image_size)  # type: ignore
            masks.append(output.squeeze(0))  # Add batch dimension
        # Stack all masks into a single tensor
        output = torch.stack(masks, dim=0)
        return output
