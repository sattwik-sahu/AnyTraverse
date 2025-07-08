import torch
from torch.nn import functional as F
from PIL import Image as PILImage
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
from anytraverse.utils.base.encoding import ImageEncoder
from anytraverse import typing as anyt
from anytraverse.helpers.device import DEVICE


class NomicImageEncoder[TImage: anyt.Image](ImageEncoder[TImage]):
    """
    Nomic image encoding model
    """

    DIM = 768

    def __init__(self, device: torch.device = DEVICE) -> None:
        super().__init__()
        self._processor = AutoImageProcessor.from_pretrained(
            "nomic-ai/nomic-embed-vision-v1.5",
            use_fast=True,
            cache_dir="data/weights/nomic",
        )
        self._model = AutoModel.from_pretrained(
            "nomic-ai/nomic-embed-vision-v1.5",
            trust_remote_code=True,
            cache_dir="data/weights/nomic",
        ).to(device=device)
        self._device = device

    def __call__(self, x: TImage | list[TImage]) -> anyt.Encoding:
        inputs = self._processor(x, return_tensors="pt").to(device=self._device)
        img_emb = self._model(**inputs).last_hidden_state
        return F.normalize(img_emb[:, 0], p=2, dim=1)
