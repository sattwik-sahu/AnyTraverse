from PIL import Image
from transformers import AutoModel, AutoProcessor
import torch
from utils.models import Model


class SigLIP(Model[Image.Image, torch.Tensor]):
    """
    Wrapper to use the SigLIP model.
    """

    def __init__(self) -> None:
        self._model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        self._processor = AutoProcessor.from_pretrained(
            "google/siglip-base-patch16-224"
        )

    def _preprocess(self, image: Image.Image):
        return self._processor(image=image, return_tensors="pt")

    def __call__(self, x: Image.Image) -> torch.Tensor:
        with torch.no_grad():
            return self._model.get_image_features(**self._preprocess(image=x))
