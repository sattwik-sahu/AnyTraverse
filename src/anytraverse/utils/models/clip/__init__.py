from typing import Callable

import clip
import clip.model
import torch
from PIL import Image
from torchvision.transforms import Compose

from anytraverse.utils.metrics.cosine import cosine_similarity
from anytraverse.utils.models import ImageEmbeddingModel


class CLIP(ImageEmbeddingModel):
    """
    Class for the CLIP image and text embedding model.
    """

    _device: torch.device
    _model: clip.model.CLIP
    _preprocess: Compose

    def __init__(self, device: torch.device = torch.device("cuda")) -> None:
        super().__init__(device=device)
        self._model, self._preprocess = clip.load("ViT-B/32", device=self._device)

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        return self._preprocess(image).unsqueeze(0).to(device=self._device)  # type: ignore

    def get_similarity(
        self,
        image1: Image.Image,
        image2: Image.Image,
        sim_func: Callable[[torch.Tensor, torch.Tensor], float] = cosine_similarity,
    ) -> float:
        """
        Gets the similarity score between two images.

        Args:
            image1 (Image.Image): First image
            image2 (Image.Image): Second image
            sim_func (Callable[[torch.Tensor, torch.Tensor], float]):
                A similarity metric calculation function which takes in
                two embeddings and returns the similarity metric.
                Default: Cosine similarity is used.

        Returns:
            float: The similarity metric between the two images' embeddings.
        """
        emb1, emb2 = (
            self._get_embedding(image=image1),
            self._get_embedding(image=image2),
        )
        if sim_func is not None:
            return sim_func(emb1, emb2)
        else:
            return cosine_similarity(x1=emb1, x2=emb2)

    def _get_embedding(self, image: Image.Image) -> torch.Tensor:
        preprocessed_image: torch.Tensor = (
            self._preprocess(img=image).unsqueeze(0).to(self._device)  # type: ignore
        )
        with torch.no_grad():
            return self._model.encode_image(image=preprocessed_image)

    def __call__(self, x: Image.Image) -> torch.Tensor:
        return self._get_embedding(image=x)
