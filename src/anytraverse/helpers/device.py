import torch


DEVICE: torch.device = torch.device(
    device="cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.mps.is_available()
    else "cpu"
)
