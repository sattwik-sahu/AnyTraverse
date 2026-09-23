"""Real-weight smoke tests for every model wrapper.

Marked ``slow``: downloads weights from the Hugging Face Hub on first use and
needs a GPU for reasonable runtimes. Excluded from the default suite::

    pytest tests/slow -m slow -o addopts=""
"""

import urllib.request
from pathlib import Path

import pytest
import torch
from PIL import Image

from anytraverse import models
from anytraverse.device import get_default_device

pytestmark = pytest.mark.slow

PROMPTS = ["road", "grass", "bush"]

# A real photograph (two cats) beats synthetic shapes: tiny detectors reliably
# fire on "cat" and ignore absurd prompts. Cached after the first download.
PHOTO_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"


@pytest.fixture(scope="module")
def photo(tmp_path_factory: pytest.TempPathFactory) -> Image.Image:
    path = Path.home() / ".cache" / "anytraverse" / "test-photo.jpg"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(PHOTO_URL, timeout=60) as response:
            path.write_bytes(response.read())
    return Image.open(path).convert("RGB")


def check_maps(maps: list, prompts: list[str], size: tuple[int, int], device: torch.device):
    """The wrapper contract every attention mapping must satisfy."""
    assert len(maps) == len(prompts)
    for m in maps:
        assert isinstance(m, torch.Tensor)
        assert tuple(m.shape) == (size[1], size[0]), m.shape
        assert m.device.type == device.type, m.device
        assert not m.requires_grad
        assert torch.isfinite(m).all()
        assert m.min() >= 0.0 and m.max() <= 1.0


def test_clipseg(photo: Image.Image) -> None:
    mapping = models.CLIPSegAttentionMapping()
    maps = mapping(photo, PROMPTS)
    check_maps(maps, PROMPTS, photo.size, mapping.device)
    # Distinct prompts give distinct maps; cache makes repeats identical.
    assert not torch.allclose(maps[0], maps[1])
    again = mapping(photo, PROMPTS)
    for a, b in zip(maps, again, strict=True):
        torch.testing.assert_close(a, b)


def test_sam3_semantic(photo: Image.Image) -> None:
    mapping = models.SAM3AttentionMapping(image_size=560)
    maps = mapping(photo, PROMPTS)
    check_maps(maps, PROMPTS, photo.size, mapping.device)


def test_sam3_instance(photo: Image.Image) -> None:
    mapping = models.SAM3AttentionMapping(mode="instance", image_size=560)
    maps = mapping(photo, ["cat", "couch"])
    check_maps(maps, ["cat", "couch"], photo.size, mapping.device)
    assert maps[0].max() > 0.5


def test_grounded_sam2(photo: Image.Image) -> None:
    mapping = models.GroundedSAM2AttentionMapping()
    maps = mapping(photo, ["cat", "couch", "purple elephant"])
    check_maps(maps, ["cat", "couch", "purple elephant"], photo.size, mapping.device)
    # The cats are detected and segmented.
    assert maps[0].max() > 0.5
    # No-detection path: an absurd threshold gives all-zero maps.
    strict = models.GroundedSAM2AttentionMapping(box_threshold=0.99)
    zeros = strict(photo, ["cat", "couch"])
    assert all(z.abs().max() == 0.0 for z in zeros)


def test_owlv2_sam2(photo: Image.Image) -> None:
    mapping = models.OWLv2SAM2AttentionMapping()
    maps = mapping(photo, ["cat", "couch", "purple elephant"])
    check_maps(maps, ["cat", "couch", "purple elephant"], photo.size, mapping.device)
    assert maps[0].max() > 0.5
    strict = models.OWLv2SAM2AttentionMapping(box_threshold=0.99)
    zeros = strict(photo, ["cat", "couch"])
    assert all(z.abs().max() == 0.0 for z in zeros)


@pytest.mark.parametrize(
    ("cls", "dim"),
    [
        (models.CLIPImageEncoder, 512),
        (models.SigLIP2ImageEncoder, 768),
        (models.DINOv2ImageEncoder, 384),
    ],
)
def test_encoders(photo: Image.Image, cls, dim: int) -> None:
    encoder = cls()
    assert encoder.dim == dim
    out = encoder(photo)
    assert tuple(out.shape) == (1, dim)
    assert not out.requires_grad and torch.isfinite(out).all()
    # Encodings are L2-normalized.
    torch.testing.assert_close(
        out.norm(dim=-1), torch.ones(1, device=out.device), atol=1e-4, rtol=0
    )
    # Batch of two works.
    assert tuple(encoder([photo, photo]).shape) == (2, dim)


def test_pipeline_end_to_end(photo: Image.Image) -> None:
    from anytraverse import build_pipeline_from_paper
    from anytraverse.state import TraversalState

    pipeline = build_pipeline_from_paper(
        init_traversability_preferences={"field": 1.0, "shed": -0.8, "rock": 0.2},
        ref_scene_similarity_threshold=0.8,
        roi_uncertainty_threshold=0.3,
    )
    state = pipeline.step(photo)
    # The synthetic image is uncertain, so the pipeline asks for help; what matters
    # is that the full state machine runs and produces valid maps either way.
    assert isinstance(state.traversal_state, TraversalState)
    assert len(pipeline.history) == 1
    assert state.traversability_map.shape == photo.size[::-1]
    assert torch.isfinite(state.traversability_map).all()
    assert torch.isfinite(state.uncertainty_map).all()
    assert 0.0 <= state.roi_traversability <= 1.0
    # A second, identical frame is fully similar to the reference.
    state2 = pipeline.step(photo)
    assert state2.ref_scene_similarity == pytest.approx(1.0, abs=1e-4)
    # The operator call path works on real models.
    pipeline.human_call("mud: -0.5")
    assert pipeline.traversability_preferences["mud"] == -0.5
    assert len(pipeline.history) == 2
    # Maps survive a trip to CPU (the ROS publishing path).
    assert state.to("cpu").traversability_map.device.type == "cpu"


def test_default_device_is_cuda() -> None:
    assert get_default_device().type == "cuda"
