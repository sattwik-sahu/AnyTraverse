import pytest
import torch

from anytraverse.models.clip import CLIPImageEncoder
from anytraverse.models.dinov2 import DINOv2ImageEncoder
from anytraverse.models.siglip2 import SigLIP2ImageEncoder

ENCODERS = [
    (CLIPImageEncoder, 8),
    (SigLIP2ImageEncoder, 6),
    (DINOv2ImageEncoder, 4),
]


@pytest.mark.parametrize(("cls", "dim"), ENCODERS)
def test_encoding_shape_dim_and_normalization(cls, dim, rgb_image, rgb_array) -> None:
    encoder = cls(device="cpu")
    assert encoder.dim == dim
    out = encoder(rgb_image)
    assert out.shape == (1, dim)
    torch.testing.assert_close(out.norm(dim=-1), torch.ones(1), atol=1e-5, rtol=0)


@pytest.mark.parametrize(("cls", "dim"), ENCODERS)
def test_batch_and_ndarray_inputs(cls, dim, rgb_image, rgb_array) -> None:
    encoder = cls(device="cpu")
    out = encoder([rgb_image, rgb_array])
    assert out.shape == (2, dim)
    single = encoder(rgb_array)
    assert single.shape == (1, dim)


@pytest.mark.parametrize("cls", [c for c, _ in ENCODERS])
def test_runs_under_inference_mode_without_grad(cls, rgb_image) -> None:
    encoder = cls(device="cpu")
    assert not encoder(rgb_image).requires_grad
