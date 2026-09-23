import pytest
import torch

from anytraverse.models.clipseg import CLIPSegAttentionMapping
from tests.conftest import HEIGHT, WIDTH
from tests.models.conftest import FROM_PRETRAINED_CALLS, FakeCLIPSegModel


@pytest.fixture
def mapping() -> CLIPSegAttentionMapping:
    return CLIPSegAttentionMapping(device="cpu")


def test_contract(mapping: CLIPSegAttentionMapping, rgb_image) -> None:
    maps = mapping(rgb_image, ["road", "bush"])
    assert len(maps) == 2
    for m in maps:
        assert m.shape == (rgb_image.size[1], rgb_image.size[0])
        assert m.dtype is torch.float32
        assert m.min() >= 0.0 and m.max() <= 1.0


def test_single_string_prompt_and_ndarray(mapping: CLIPSegAttentionMapping, rgb_array) -> None:
    maps = mapping(rgb_array, "road")
    assert len(maps) == 1
    assert maps[0].shape == (HEIGHT, WIDTH)


def test_prompts_are_distinguishable(mapping: CLIPSegAttentionMapping, rgb_image) -> None:
    road, bush = mapping(rgb_image, ["road", "bush"])
    assert not torch.allclose(road, bush)


def test_image_encoded_once_and_prompts_cached(mapping: CLIPSegAttentionMapping, rgb_image) -> None:
    mapping(rgb_image, ["road", "bush"])
    assert FakeCLIPSegModel.vision_calls == 1
    assert FakeCLIPSegModel.cond_calls == 1
    mapping(rgb_image, ["road", "bush"])
    assert FakeCLIPSegModel.vision_calls == 2
    assert FakeCLIPSegModel.cond_calls == 1  # cache hit
    mapping(rgb_image, ["road", "mud"])
    assert FakeCLIPSegModel.cond_calls == 2


def test_empty_prompts_raise(mapping: CLIPSegAttentionMapping, rgb_image) -> None:
    with pytest.raises(ValueError, match="At least one prompt"):
        mapping(rgb_image, [])


def test_prompt_cache_evicts_when_full(mapping: CLIPSegAttentionMapping, rgb_image) -> None:
    for i in range(65):
        mapping(rgb_image, f"prompt-{i}")
    assert FakeCLIPSegModel.cond_calls == 65
    maps = mapping(rgb_image, "road")
    assert len(maps) == 1


def test_model_id_dtype_and_cache_forwarded() -> None:
    CLIPSegAttentionMapping(model_id="custom/clipseg", device="cpu", cache_dir="/tmp/hf")
    names = {(name, mid) for name, mid, _ in FROM_PRETRAINED_CALLS}
    assert ("FakeCLIPSegProcessor", "custom/clipseg") in names
    assert ("FakeCLIPSegModel", "custom/clipseg") in names
    for _, _, kwargs in FROM_PRETRAINED_CALLS:
        assert kwargs["cache_dir"] == "/tmp/hf"
