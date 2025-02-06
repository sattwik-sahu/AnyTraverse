import numpy as np
import torch


def iou_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the Intersection over Union (IoU) score.

    Args:
        y_true (np.ndarray): Ground truth mask.
        y_pred (np.ndarray): Predicted mask.

    Returns:
        float: IoU score.
    """
    num = np.sum(y_true & y_pred, dtype=float)
    den = np.sum(y_true | y_pred, dtype=float)
    if den == 0:
        return 1.0
    return num / den


def iou_torch(y_true: torch.BoolTensor, y_pred: torch.BoolTensor) -> float:
    """
    Computes the IoU score between two `torch.BoolTensor` masks.

    Args:
        y_true (torch.BoolTensor): The ground truth mask.
        y_pred (torch.BoolTensor): The predicted mask.

    Returns:
        float: The IoU score.
    """
    return float(torch.sum(y_true & y_pred) / torch.sum(y_true | y_pred))
