import numpy as np
from matplotlib import font_manager, pyplot as plt
from matplotlib.axes import Axes
from matplotlib.transforms import Bbox
from matplotlib.figure import Figure
from PIL import Image
from numpy import typing as npt
from typing import Dict, Any, Tuple, List


def plot_anytraverse_output(
    image: Image.Image,
    mask_true: np.ndarray,
    mask_pred: np.ndarray,
    params: Dict[str, Any],
    fig: Figure | None = None,
    ax: List[Axes] | None = None,
) -> Tuple[Figure, List[Axes]]:
    """
    Plots the output of the AnyTraverse pipeline. There are three subplots:
    - The original image
    - The true mask
    - The predicted mask

    There are also some parameters shown in the plot, passed in `params`.

    Args:
        image (Image.Image): The original image.
        mask_true (np.ndarray): The ground truth mask.
        mask_pred (np.ndarray): The predicted mask.
        params (Dict[str, Any]): The parameters to show in the plot.

    Returns:
        Tuple[Figure, List[Axes]]: The figure and axes of the plot.
    """
    if fig is None and ax is None:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    if fig is not None and ax is not None:
        ax[0].imshow(image)
        ax[0].set_title("Input Image")
        ax[0].set_axis_off()

        ax[1].imshow(image)
        ax[1].imshow(mask_true, alpha=0.3, cmap="inferno")
        ax[1].set_title("Ground Truth Mask")
        ax[1].set_axis_off()

        ax[2].imshow(image)
        ax[2].imshow(mask_pred, alpha=0.3, cmap="inferno")
        ax[2].set_title("Predicted Mask")
        ax[2].set_axis_off()

        # Add the parameters to the plot
        cell_text = [[f"{k}", f"{v}"] for k, v in params.items()]
        table = ax[3].table(
            cellText=cell_text,
            colLabels=["Parameter", "Value"],
            cellLoc="center",
            loc="center",
            colColours=["#f0f0f0", "#f0f0f0"],
            colWidths=[0.3, 0.7],
            bbox=Bbox.from_bounds(0.35, 0.1, 0.3, 0.8),
        )
        for (i, j), cell in table.get_celld().items():
            cell.set_height(0.2)

        # Customize table appearance
        table.auto_set_font_size(False)
        table.set_fontsize(12)

    return fig, ax  # type: ignore
