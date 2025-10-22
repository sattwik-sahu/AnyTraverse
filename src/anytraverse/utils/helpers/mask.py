import torch


def convert_to_seg_mask(mask: torch.Tensor, threshold: float) -> torch.Tensor:
    return mask > threshold
