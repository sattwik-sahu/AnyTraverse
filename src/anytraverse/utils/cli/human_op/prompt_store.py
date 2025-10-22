from typing import List, Tuple, TypedDict

import numpy as np
import torch
from numpy import typing as npt
from PIL import Image
import pickle
from anytraverse.config.utils import WeightedPrompt
from anytraverse.utils.models.clip import CLIP
from pathlib import Path
from anytraverse.utils.models import ImageEmbeddingModel
from anytraverse.utils.cli.human_op.models import (
    ImageEmbeddings,
    HistoryPickle,
    SceneWeightedPrompt,
)


class ScenePromptStoreManager:
    _store: List[SceneWeightedPrompt]
    _image_embedding_model: ImageEmbeddingModel

    def __init__(
        self, image_embedding_model: ImageEmbeddingModel | None = None
    ) -> None:
        self._store = []
        self._image_embedding_model = image_embedding_model or CLIP(
            device=torch.device("cuda")
        )

    @property
    def scene_embeddings(self) -> torch.Tensor:
        return torch.cat(
            [sp["scene"]["ref_frame_embedding"] for sp in self._store], dim=0
        )

    def is_empty(self) -> bool:
        return len(self._store) == 0

    def add_scene_prompt(self, scene_prompt: SceneWeightedPrompt) -> None:
        if len(scene_prompt["prompts"]) > 0:
            self._store.append(scene_prompt)

    def _get_similarities(self, frame: Image.Image) -> torch.Tensor:
        frame_embedding: torch.Tensor = self._image_embedding_model(x=frame)
        return torch.cosine_similarity(x1=frame_embedding, x2=self.scene_embeddings)

    def get_best_match(self, frame: Image.Image) -> Tuple[SceneWeightedPrompt, float]:
        similarities: npt.NDArray = self._get_similarities(frame=frame).cpu().numpy()
        best_match_inx: int = int(np.argmax(similarities))
        best_match: SceneWeightedPrompt = self._store[best_match_inx]
        return best_match, similarities[best_match_inx]


def save_store(
    image_embeddings: ImageEmbeddings,
    scene_prompts_store_manager: ScenePromptStoreManager,
    filepath: Path,
) -> HistoryPickle:
    """
    Saves the store to the given filepath. The image_embeddings object is just so that the whole pytorch model is not
    pickled, instead only the scene reference frames and the corresponding prompts are pickled along with the info
    of which image embeddings were used.

    Args:
        image_embeddings (ImageEmbeddings): The enum specifying whcih image embeddings were used with the store.
        scene_prompts_store_manager (ScenePromptsStoreManager): The scene prompts store manager which has the store to pickle.
        filepath (Path): The path to the pickled file.

    Returns:
        HistoryPickle: The history object that was stored in the pickle file.
    """
    # Create the history object to pickle
    hist_obj: HistoryPickle = HistoryPickle(
        image_embeddings=image_embeddings,
        scene_prompts_store=scene_prompts_store_manager._store,
    )

    # Open the file to write in binary mode ("wb")
    with open(filepath, "wb") as f:
        # Write the pickle object to the file
        pickle.dump(obj=hist_obj, file=f)

    return hist_obj
