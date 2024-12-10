import numpy as np
from numpy import typing as npt


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
