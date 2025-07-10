import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoImageProcessor
from anytraverse.utils.base.encoding import HuggingFaceImageEncoder
from anytraverse import typing as anyt
from anytraverse.helpers.device import DEVICE


class NomicImageEncoder[TImage: anyt.Image](HuggingFaceImageEncoder[TImage]):
    """
    Nomic image encoding model
    """

    DIM = 768

    def __init__(self, device: torch.device = DEVICE) -> None:
        super().__init__(
            model_name="nomic-ai/nomic-embed-vision-v1.5",
            device=device,
            cache_dir="data/weights/nomic",
        )
        self._processor = AutoImageProcessor.from_pretrained(
            self._model_name,
            use_fast=self._use_fast_processor,
            cache_dir=self._cache_dir,
        )
        self._model = AutoModel.from_pretrained(
            self._model_name,
            trust_remote_code=True,
            cache_dir=self._cache_dir,
            device_map=self._device,
        )

    def __call__(self, x: TImage | list[TImage]) -> anyt.Encoding:
        inputs = self._processor(x, return_tensors="pt").to(device=self._device)
        img_emb = self._model(**inputs).last_hidden_state
        return F.normalize(img_emb[:, 0], p=2, dim=1)
