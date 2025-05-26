from typing import Tuple
from matplotlib.patches import Rectangle
from matplotlib.axes import Axes
from anytraverse.utils.metrics.roi import ROI_Checker
from PIL import Image
from anytraverse.utils.viz.common import overlay_mask_cv2
import torch


def plot_roi_on_axes(
    ax: Axes,
    x_start: float,
    x_end: float,
    y_start: float,
    y_end: float,
    edgecolor: str = "lightgreen",
    linewidth: float = 2.0,
    linestyle: str = "-",
    alpha: float = 1.0,
    label: str = "rect",
) -> None:
    """
    Annotates a rectangle on an image using a matplotlib axis object.

    Args:
        ax (plt.Axes): The matplotlib axis object containing the image.
        x_start (float): The starting x-coordinate of the rectangle.
        x_end (float): The ending x-coordinate of the rectangle.
        y_start (float): The starting y-coordinate of the rectangle.
        y_end (float): The ending y-coordinate of the rectangle.
        edgecolor (str, optional): The color of the rectangle's border. Defaults to "lightgreen".
        linewidth (float, optional): The width of the rectangle's border. Defaults to 2.0.
        linestyle (str, optional): The style of the rectangle's border. Defaults to "-" (solid line).
        alpha (float, optional): The transparency of the rectangle's border. Defaults to 1.0 (opaque).

    Returns:
        None
    """
    # Calculate the width and height of the rectangle
    width = x_end - x_start
    height = y_end - y_start

    # Create a rectangle patch with no fill
    rect = Rectangle(
        (x_start, y_start),  # Bottom-left corner coordinates
        width,  # Width of the rectangle
        height,  # Height of the rectangle
        linewidth=linewidth,  # Border width
        edgecolor=edgecolor,  # Border color
        facecolor="none",  # No fill
        linestyle=linestyle,  # Border style
        alpha=alpha,  # Transparency
        label=label,
    )

    # Add the rectangle to the axis
    ax.add_patch(rect)


def plot_image_seg_roi(
    ax: Axes,
    image: Image.Image,
    mask: torch.Tensor,
    threshold: float,
    roi: ROI_Checker,
    title: str | None = None,
    cmap: str = "jet",
    edgecolor: str = "cyan",
    linewidth: int = 2,
    mask_alpha: float = 0.25,
    roi_alpha: float = 0.9,
    msg: str = "Mask",
    color: Tuple[int, int, int] = (0, 255, 0),
):
    # Plot Image
    # ax.imshow(image)

    # Calculate ROI traversability
    thresh_mask = mask.cpu() > threshold

    # Plot segmentation mask
    # ax.imshow(thresh_mask.cpu().numpy() > threshold, cmap=cmap, alpha=mask_alpha)
    ax.imshow(
        overlay_mask_cv2(image=image, mask=thresh_mask, alpha=mask_alpha, color=color)
    )

    # Plot ROI box
    (xs, xe), (ys, ye) = roi._get_roi_bounds_in_pixels(mask=mask)
    plot_roi_on_axes(
        ax=ax,
        x_start=xs,
        x_end=xe,
        y_start=ys,
        y_end=ye,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=roi_alpha,
    )

    # Show ROI Traversability percentage
    trav_roi: float = roi.trav_area(mask=thresh_mask)
    ax.text(
        x=30,
        y=100,
        s=f"{msg} in ROI = {trav_roi * 100 :.2f}%",
        color=edgecolor,
        backgroundcolor="#0b2e3cde",
        family="monospace",
        size="large",
        weight="bold",
    )

    # Show title on axes
    if title is not None:
        ax.set_title(label=title)
