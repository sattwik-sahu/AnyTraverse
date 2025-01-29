from typing import List, Tuple, TypedDict

import numpy as np
import torch
from numpy import typing as npt
from PIL import Image

from config.utils import WeightedPrompt
from utils.models.clip import CLIP

WeightedPromptList = List[WeightedPrompt]


class Scene(TypedDict):
    ref_frame: Image.Image
    ref_frame_embedding: torch.Tensor


class SceneWeightedPrompt(TypedDict):
    scene: Scene
    prompts: WeightedPromptList


class ScenePromptStoreManager:
    _store: List[SceneWeightedPrompt]
    _clip: CLIP

    def __init__(self) -> None:
        self._store = []
        self._clip = CLIP(device=torch.device("cuda"))

    @property
    def scene_embeddings(self) -> torch.Tensor:
        return torch.cat(
            [sp["scene"]["ref_frame_embedding"] for sp in self._store], dim=0
        )

    def add_scene_prompt(self, scene_prompt: SceneWeightedPrompt) -> None:
        if len(scene_prompt["prompts"]) > 0:
            self._store.append(scene_prompt)

    def _get_similarities(self, frame: Image.Image) -> torch.Tensor:
        frame_embedding: torch.Tensor = self._clip(image=frame)
        return torch.cosine_similarity(x1=frame_embedding, x2=self.scene_embeddings)

    def get_best_match(self, frame: Image.Image) -> Tuple[SceneWeightedPrompt, float]:
        similarities: npt.NDArray = self._get_similarities(frame=frame).cpu().numpy()
        best_match_inx: int = int(np.argmax(similarities))
        best_match: SceneWeightedPrompt = self._store[best_match_inx]
        return best_match, similarities[best_match_inx]
