"""Hardware device and precision selection."""

import torch


def get_default_device() -> torch.device:
    """
    Picks the best available accelerator.

    The preference order is ``cuda`` > ``mps`` > ``cpu``. The check happens
    every time the function is called, so environment variables such as
    ``CUDA_VISIBLE_DEVICES`` are respected even when set after import.

    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_default_dtype(device: torch.device) -> torch.dtype:
    """
    Picks a sensible inference precision for a device.

    Half precision is used on CUDA where it roughly halves memory and
    latency; ``float32`` is used everywhere else since ``float16`` support
    on CPU and MPS is patchy.

    Args:
        device (torch.device): The device the model will run on.

    Returns:
        torch.dtype: ``torch.float16`` on CUDA, ``torch.float32`` otherwise.
    """
    return torch.float16 if device.type == "cuda" else torch.float32


def resolve_device(device: str | torch.device | None) -> torch.device:
    """
    Normalizes a user-supplied device argument.

    Args:
        device (str | torch.device | None): A device string such as ``"cuda:0"``,
            a ``torch.device``, or ``None`` to auto-detect.

    Returns:
        torch.device: The resolved device.
    """
    if device is None:
        return get_default_device()
    return torch.device(device)


__all__ = ["get_default_device", "get_default_dtype", "resolve_device"]
