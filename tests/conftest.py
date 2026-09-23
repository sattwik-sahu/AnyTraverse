"""Shared fixtures: tiny images and deterministic fake models that need no weights."""

from collections.abc import Sequence
from typing import override

import numpy as np
import pytest
import torch
from PIL import Image as PILImage

from anytraverse import typing as anyt
from anytraverse.interfaces import ImageEncoder, PromptAttentionMapping

HEIGHT, WIDTH = 24, 32


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture
def rgb_array() -> np.ndarray:
    """A ``(H, W, 3)`` uint8 image with a bright bottom half and a dark top half."""
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[HEIGHT // 2 :] = 200
    return image


@pytest.fixture
def rgb_image(rgb_array: np.ndarray) -> PILImage.Image:
    return PILImage.fromarray(rgb_array)


@pytest.fixture
def prefs() -> anyt.TraversabilityPreferences:
    return {"road": 1.0, "bush": -0.8, "rock": 0.45}


@pytest.fixture
def maps() -> list[anyt.PromptAttentionMap]:
    """Three attention maps with clearly separated regions."""
    road = torch.zeros(HEIGHT, WIDTH)
    road[HEIGHT // 2 :] = 0.9
    bush = torch.zeros(HEIGHT, WIDTH)
    bush[: HEIGHT // 2, : WIDTH // 2] = 0.8
    rock = torch.full((HEIGHT, WIDTH), 0.2)
    return [road, bush, rock]


class FakeAttentionMapping(PromptAttentionMapping):
    """Returns a fixed score per prompt over the whole image; records the prompts it saw."""

    def __init__(self, scores: dict[str, float] | None = None, default: float = 0.5) -> None:
        self.scores = scores or {}
        self.default = default
        self.calls: list[list[str]] = []

    @override
    def __call__(
        self, x: anyt.Image, prompts: anyt.Prompt | Sequence[anyt.Prompt]
    ) -> list[anyt.PromptAttentionMap]:
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        self.calls.append(prompts)
        height, width = x.shape[:2] if isinstance(x, np.ndarray) else x.size[::-1]
        return [
            torch.full((height, width), self.scores.get(p, self.default), dtype=torch.float32)
            for p in prompts
        ]


class FakeEncoder(ImageEncoder):
    """Returns a queued list of encodings, one per call, then repeats the last one."""

    dim = 4

    def __init__(self, encodings: Sequence[torch.Tensor] | None = None) -> None:
        self.queue = list(encodings or [torch.tensor([[1.0, 0.0, 0.0, 0.0]])])
        self.calls = 0

    @override
    def __call__(self, x: anyt.Image | Sequence[anyt.Image]) -> anyt.Encoding:
        self.calls += 1
        index = min(self.calls - 1, len(self.queue) - 1)
        return self.queue[index].clone()


@pytest.fixture
def fake_attention() -> FakeAttentionMapping:
    return FakeAttentionMapping({"road": 0.9, "bush": 0.1, "rock": 0.3})


@pytest.fixture
def fake_encoder() -> FakeEncoder:
    return FakeEncoder()


def unit(*values: float) -> torch.Tensor:
    """A ``(1, 4)`` L2-normalized encoding built from up to four components."""
    vec = torch.zeros(1, 4)
    vec[0, : len(values)] = torch.tensor(values)
    return vec / vec.norm()
