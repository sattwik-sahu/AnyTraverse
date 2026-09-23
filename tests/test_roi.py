import numpy as np
import pytest
import torch

from anytraverse.roi import RegionOfInterest


@pytest.fixture
def roi() -> RegionOfInterest:
    return RegionOfInterest(x_bounds=(0.25, 0.75), y_bounds=(0.5, 1.0))


def test_pixel_bounds_are_inclusive_and_clamped(roi: RegionOfInterest) -> None:
    assert roi.pixel_bounds(height=20, width=40) == ((10, 10), (30, 19))


def test_extract_torch(roi: RegionOfInterest) -> None:
    mat = torch.arange(20 * 40, dtype=torch.float32).reshape(20, 40)
    crop, start, end = roi.extract(mat)
    assert (start, end) == ((10, 10), (30, 19))
    assert crop.shape == (10, 21)
    torch.testing.assert_close(crop, mat[10:20, 10:31])


def test_extract_numpy(roi: RegionOfInterest) -> None:
    mat = np.arange(20 * 40).reshape(20, 40)
    crop, _, _ = roi.extract(mat)
    assert isinstance(crop, np.ndarray)
    assert crop.shape == (10, 21)


def test_properties(roi: RegionOfInterest) -> None:
    assert roi.x_bounds == (0.25, 0.75)
    assert roi.y_bounds == (0.5, 1.0)


@pytest.mark.parametrize(
    ("x", "y"),
    [
        ((0.5, 0.5), (0.0, 1.0)),
        ((0.7, 0.3), (0.0, 1.0)),
        ((0.0, 1.0), (-0.1, 1.0)),
        ((0.0, 1.1), (0.0, 1.0)),
    ],
)
def test_invalid_bounds_raise(x, y) -> None:
    with pytest.raises(ValueError, match="must satisfy"):
        RegionOfInterest(x_bounds=x, y_bounds=y)
