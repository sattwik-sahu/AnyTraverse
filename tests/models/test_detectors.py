import pytest
import torch
from PIL import Image as PILImage

from anytraverse.models._sam2 import Sam2BoxSegmenter
from anytraverse.models.grounded_sam2 import GroundedSAM2AttentionMapping
from anytraverse.models.owlv2_sam2 import OWLv2SAM2AttentionMapping
from tests.models.conftest import FakeSam2Model


@pytest.fixture
def image() -> PILImage.Image:
    return PILImage.new("RGB", (32, 24))


class TestSam2BoxSegmenter:
    def test_empty_boxes_give_zeros(self, image) -> None:
        segmenter = Sam2BoxSegmenter(device="cpu")
        out = segmenter(image, [[], []])
        assert out.shape == (2, 24, 32)
        assert out.abs().max() == 0.0

    def test_single_forward_for_all_boxes(self, image) -> None:
        segmenter = Sam2BoxSegmenter(device="cpu")
        out = segmenter(image, [[(1, 1, 10, 10)], [(2, 2, 8, 8), (3, 3, 9, 9)]])
        assert FakeSam2Model.forward_calls == 1
        assert out.shape == (2, 24, 32)
        assert (out[0].max() > 0.5) and (out[1].max() > 0.5)
        assert out.min() >= 0.0 and out.max() <= 1.0


class TestGroundedSAM2:
    def test_groups_boxes_by_prompt(self, image) -> None:
        mapping = GroundedSAM2AttentionMapping(device="cpu")
        maps = mapping(image, ["road", "grass"])
        assert len(maps) == 2
        assert all(m.shape == (24, 32) for m in maps)
        # The fake detector fires one box for each prompt; both maps are non-empty.
        assert all(m.max() > 0.5 for m in maps)

    def test_single_prompt(self, image) -> None:
        (m,) = GroundedSAM2AttentionMapping(device="cpu")(image, "road")
        assert m.shape == (24, 32)
        assert m.max() > 0.5

    def test_no_detections_give_zeros(self, image) -> None:
        mapping = GroundedSAM2AttentionMapping(box_threshold=0.99, device="cpu")
        maps = mapping(image, ["road", "grass"])
        assert all(m.abs().max() == 0.0 for m in maps)

    def test_token_masks_skip_separators(self, image) -> None:
        mapping = GroundedSAM2AttentionMapping(device="cpu")
        ids = torch.tensor([101, 2001, 1012, 2002, 102])
        masks = mapping._prompt_token_masks(ids, 2)
        assert masks.tolist() == [
            [False, True, False, False, False],
            [False, False, False, True, False],
        ]

    def test_token_masks_ignore_trailing_tokens(self, image) -> None:
        # More separators than prompts: tokens after the last separator are dropped.
        mapping = GroundedSAM2AttentionMapping(device="cpu")
        ids = torch.tensor([101, 2001, 1012, 1012, 2002, 102])
        masks = mapping._prompt_token_masks(ids, 1)
        assert masks.tolist() == [[False, True, False, False, False, False]]


class TestOWLv2SAM2:
    def test_boxes_grouped_and_clamped(self, image) -> None:
        (m0, m1) = OWLv2SAM2AttentionMapping(device="cpu")(image, ["road", "grass"])
        assert m0.shape == m1.shape == (24, 32)
        assert m0.max() > 0.5  # two kept boxes for prompt 0
        assert m1.abs().max() == 0.0  # the only box for prompt 1 is below threshold

    def test_low_threshold_keeps_weak_box(self, image) -> None:
        (_m0, m1) = OWLv2SAM2AttentionMapping(box_threshold=0.01, device="cpu")(
            image, ["road", "grass"]
        )
        assert m1.max() > 0.5
