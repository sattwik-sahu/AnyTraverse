import pytest
import torch

from anytraverse.device import get_default_device, get_default_dtype, resolve_device


def test_default_device_prefers_cuda_then_mps_then_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert get_default_device().type == "cuda"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert get_default_device().type == "mps"

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert get_default_device().type == "cpu"


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        ("cuda", torch.float16),
        ("cuda:1", torch.float16),
        ("cpu", torch.float32),
        ("mps", torch.float32),
    ],
)
def test_default_dtype(device: str, expected: torch.dtype) -> None:
    assert get_default_dtype(torch.device(device)) is expected


def test_resolve_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device(None) == torch.device("cpu")
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device(torch.device("cpu")) == torch.device("cpu")
