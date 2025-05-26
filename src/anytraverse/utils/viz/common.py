import cv2
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import List, Tuple
from PIL import Image
import torch
import numpy as np


def overlay_image(
    image: Image.Image,
    overlay: torch.Tensor | np.ndarray,
    overlay_title: str = "Overlay",
    cmap: str = "jet",
    alpha: float = 0.5,
    figsize: Tuple[int, int] = (16, 12),
    threshold: float = 0.5,
) -> Tuple[Figure, List[Axes]]:
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=figsize)

    ax[0].set_title(f"Image {'x'.join(map(str, image.size))}")
    ax[0].imshow(image)

    ax[1].set_title(overlay_title)
    ax[1].imshow(image)
    ax[1].imshow(overlay, alpha=alpha, cmap=cmap)

    ax[2].set_title(f"{overlay_title} Thresholded ({threshold})")
    ax[2].imshow(image)
    ax[2].imshow(overlay > threshold, alpha=alpha, cmap=cmap)

    return fig, ax


def overlay_mask_cv2(
    image: Image.Image, mask: torch.Tensor, color=(0, 255, 0), alpha=0.5
):
    """
    Overlay a segmentation mask on an image using OpenCV.

    :param image: PIL image
    :param mask: torch.Tensor of shape (H, W) with boolean values
    :param color: RGB tuple for the mask color
    :param alpha: Alpha blending value (0.0 - 1.0)
    :return: PIL image with the mask overlay
    """
    # Convert image to OpenCV format (numpy array)
    image_cv2 = np.array(image)
    if image_cv2.shape[-1] == 4:  # Convert RGBA to BGR if needed
        image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGBA2RGB)

    # Convert mask to numpy array (binary)
    mask_np = mask.cpu().numpy().astype(np.uint8)

    # Create a colored overlay
    overlay = np.zeros_like(image_cv2, dtype=np.uint8)
    overlay[mask_np == 1] = color  # Apply color where mask is True

    # Blend images using alpha
    result = cv2.addWeighted(image_cv2, 1 - alpha, overlay, alpha, 0)

    # Convert back to PIL for compatibility
    return Image.fromarray(result)
