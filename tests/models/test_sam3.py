import pytest

from anytraverse.models.sam3 import SAM3AttentionMapping
from tests.models.conftest import FakeSam3Config, FakeSam3Model


@pytest.fixture
def mapping() -> SAM3AttentionMapping:
    return SAM3AttentionMapping(device="cpu")


def test_semantic_contract(mapping: SAM3AttentionMapping, rgb_image) -> None:
    maps = mapping(rgb_image, ["road", "grass"])
    assert len(maps) == 2
    for m in maps:
        assert m.shape == (rgb_image.size[1], rgb_image.size[0])
        assert m.min() >= 0.0 and m.max() <= 1.0
    # Fake semantic head is hot in the bottom half, cold in the top half.
    assert maps[0][-1, 0] > 0.9
    assert maps[0][0, 0] < 0.1


def test_vision_encoded_once_text_cached(mapping: SAM3AttentionMapping, rgb_image) -> None:
    mapping(rgb_image, ["road", "grass"])
    assert FakeSam3Model.vision_calls == 1
    assert FakeSam3Model.text_calls == 2
    assert FakeSam3Model.forward_calls == 2
    mapping(rgb_image, ["road", "grass"])
    assert FakeSam3Model.vision_calls == 2
    assert FakeSam3Model.text_calls == 2  # cache hits
    assert FakeSam3Model.forward_calls == 4


def test_instance_mode_takes_left_half(mapping: SAM3AttentionMapping, rgb_image) -> None:
    # "rock" has an odd fake token id, so the fake model returns a confident instance;
    # even ids exercise the no-instances path instead (see the strict-threshold test).
    instance = SAM3AttentionMapping(mode="instance", device="cpu")
    (m,) = instance(rgb_image, "rock")
    assert m.shape == (rgb_image.size[1], rgb_image.size[0])
    assert m[:, : m.shape[1] // 2].mean() > m[:, m.shape[1] // 2 :].mean()


def test_instance_mode_zeros_when_nothing_passes_threshold(rgb_image) -> None:
    strict = SAM3AttentionMapping(mode="instance", instance_threshold=0.9999, device="cpu")
    (m,) = strict(rgb_image, "road")
    assert m.abs().max() == 0.0


def test_prompt_cache_evicts_when_full(mapping: SAM3AttentionMapping, rgb_image) -> None:
    for i in range(65):
        mapping(rgb_image, f"prompt-{i}")
    assert FakeSam3Model.text_calls == 65
    assert len(mapping(rgb_image, "road")) == 1


def test_invalid_mode_raises() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        SAM3AttentionMapping(mode="boxes", device="cpu")  # type: ignore[arg-type]


def test_image_size_configures_model_and_processor() -> None:
    SAM3AttentionMapping(image_size=560, device="cpu")
    assert FakeSam3Config.calls == ["facebook/sam3"]


def test_single_string_and_ndarray_inputs(mapping: SAM3AttentionMapping, rgb_array) -> None:
    maps = mapping(rgb_array, "road")
    assert len(maps) == 1 and maps[0].shape == rgb_array.shape[:2]
