import torch


def cosine_similarity(x1: torch.Tensor, x2: torch.Tensor) -> float:
    return torch.cosine_similarity(x1=x1, x2=x2).cpu().item()
