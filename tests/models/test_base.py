import numpy as np
import pytest
import torch
from PIL import Image as PILImage

from anytraverse.models._base import (
    HuggingFaceModel,
    as_prompt_list,
    resize_maps,
    to_pil,
    to_pil_list,
)
from tests.models.conftest import (
    FROM_PRETRAINED_CALLS,
    FakeCLIPSegProcessor,
    FakeImageProcessor,
    FakeModule,
)


def test_to_pil_converts_rgb_array() -> None:
    image = to_pil(np.zeros((8, 10, 3), dtype=np.uint8))
    assert isinstance(image, PILImage.Image)
    assert image.size == (10, 8)
    assert image.mode == "RGB"


def test_to_pil_passes_rgb_through() -> None:
    image = PILImage.new("RGB", (10, 8))
    assert to_pil(image) is image


def test_to_pil_converts_grayscale() -> None:
    assert to_pil(PILImage.new("L", (10, 8))).mode == "RGB"


def test_to_pil_list_single_and_sequence(rgb_image, rgb_array) -> None:
    assert to_pil_list(rgb_image) == [rgb_image]
    assert len(to_pil_list([rgb_image, rgb_array])) == 2
    assert all(isinstance(i, PILImage.Image) for i in to_pil_list((rgb_array,)))


def test_resize_maps_noop_when_same_size() -> None:
    maps = torch.rand(3, 8, 10)
    assert resize_maps(maps, (8, 10)) is maps


def test_resize_maps_rescales() -> None:
    out = resize_maps(torch.ones(2, 16, 16), (8, 4))
    assert out.shape == (2, 8, 4)
    assert out.min() >= 0.0


def test_as_prompt_list() -> None:
    assert as_prompt_list("road") == ["road"]
    assert as_prompt_list(("a", "b")) == ["a", "b"]
    with pytest.raises(ValueError, match="At least one prompt"):
        as_prompt_list([])


def test_require_transformers_returns_module() -> None:
    import anytraverse.models._base as base

    # The fixture patches the module attribute, which is what the wrappers call.
    transformers = base.require_transformers()
    assert transformers.AutoProcessor is FakeCLIPSegProcessor
    assert transformers.AutoImageProcessor is FakeImageProcessor


def test_require_transformers_real_import() -> None:
    """Exercises the real lazy import (unpatched) against the installed transformers."""
    import importlib

    import anytraverse.models._base as base

    namespace = dict(base.__dict__)
    try:
        reloaded = importlib.reload(base)
        assert reloaded.require_transformers().AutoProcessor is not None
    finally:
        base.__dict__.clear()
        base.__dict__.update(namespace)


def test_huggingface_model_defaults_to_cpu_dtype() -> None:
    model = HuggingFaceModel("some-id", device="cpu")
    assert model.device == torch.device("cpu")
    assert model.dtype is torch.float32
    assert model.cache_dir is None
    assert model.compile is False


def test_load_model_forwards_options_and_evals() -> None:
    model = HuggingFaceModel("some-id", device="cpu", dtype=torch.float32, cache_dir="/tmp/hf")
    assert model._load_model(FakeModule).training is False
    assert ("FakeModule", "some-id", {"dtype": torch.float32, "cache_dir": "/tmp/hf"}) in [
        (name, mid, kwargs) for name, mid, kwargs in FROM_PRETRAINED_CALLS
    ]


def test_load_model_compile_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(torch, "compile", lambda m, **kw: calls.append(kw) or m)
    model = HuggingFaceModel("some-id", device="cpu", compile=True)
    model._load_model(FakeModule)
    assert calls == [{"mode": "reduce-overhead"}]


def test_load_model_skips_compile_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch, "compile", lambda m, **kw: pytest.fail("should not compile"))
    HuggingFaceModel("some-id", device="cpu")._load_model(FakeModule)


def test_to_device_casts_floats_only() -> None:
    from tests.models.conftest import FakeBatch

    model = HuggingFaceModel("some-id", device="cpu", dtype=torch.float32)
    batch = FakeBatch(a=torch.ones(2), b=torch.ones(2, dtype=torch.long), c="text")
    moved = model._to_device(batch)
    assert moved["a"].dtype is torch.float32
    assert moved["b"].dtype is torch.long
    assert moved["c"] == "text"
