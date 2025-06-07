import torch
import cv2
from matplotlib import pyplot as plt
import numpy as np


def convert_to_seg_mask(mask: torch.Tensor, threshold: float) -> torch.Tensor:
    return mask > threshold


def torch_tensor_to_cv_image(
    tensor: torch.Tensor, cmap_name: str = "plasma"
) -> np.ndarray:
    """
    Convert a 2D torch.Tensor (values in [0, 1]) to a BGR image using a matplotlib colormap.

    Args:
        tensor (torch.Tensor): 2D tensor with float values in [0, 1].
        cmap_name (str): Name of matplotlib colormap (e.g., 'plasma', 'viridis').

    Returns:
        np.ndarray: BGR image (uint8) suitable for OpenCV display or saving.
    """
    if tensor.ndim != 2:
        raise ValueError("Only 2D tensors are supported.")

    # Convert to numpy and clamp to [0, 1]
    data = tensor.detach().cpu().numpy()
    data = np.clip(data, 0, 1)

    # Convert to uint8
    data_uint8 = (255 * data).astype(np.uint8)

    # Apply colormap using matplotlib
    cmap = plt.get_cmap(cmap_name)
    colored = cmap(data_uint8 / 255.0)[..., :3]  # (H, W, 3), RGB
    colored = (colored * 255).astype(np.uint8)  # to uint8

    # Convert RGB to BGR for OpenCV
    colored_bgr = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
    return colored_bgr
