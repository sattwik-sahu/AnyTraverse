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
    threshold: float = 0.5
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
