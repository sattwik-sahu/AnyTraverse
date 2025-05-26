from PIL import Image
from transformers import AutoModel, AutoProcessor
import torch
from anytraverse.utils.models import ImageEmbeddingModel


class SigLIP(ImageEmbeddingModel):
    """
    Wrapper to use the SigLIP model.
    """

    def __init__(self, device: torch.device) -> None:
        super().__init__(device=device)
        torch.set_default_device(device=self._device)
        self._model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(
            device=self._device
        )
        self._processor = AutoProcessor.from_pretrained(
            "google/siglip-base-patch16-224"
        )

    def _preprocess(self, image: Image.Image):
        return self._processor(images=image, return_tensors="pt")

    def __call__(self, x: Image.Image) -> torch.Tensor:
        with torch.no_grad():
            return self._model.get_image_features(**self._preprocess(image=x)).to(
                device=self._device
            )
