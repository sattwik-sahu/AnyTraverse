from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from PIL.Image import Image
import torch
from typing import Tuple, List


def plot_image_and_depthmap(
    image: Image,
    depth: torch.Tensor,
    image_title: str = "Image",
    depth_title: str = "Depth Map",
) -> Tuple[Figure, List[Axes]]:
    """Creates a side-by-side visualization of an image and its corresponding depth map.

    This function generates a figure with two subplots: one showing the original image
    and another showing the depth map with a colorbar. The depth map is displayed using
    the 'jet' colormap for better depth perception.

    Args:
        image (PIL.Image.Image): Input RGB or grayscale image to be displayed
        depth (torch.Tensor): Depth map tensor corresponding to the image. Should be a
            2D tensor with values representing depths
        image_title (str, optional): Title for the image subplot. Defaults to "Image"
        depth_title (str, optional): Title for the depth map subplot. Defaults to "Depth Map"

    Returns:
        Tuple[matplotlib.figure.Figure, List[matplotlib.axes.Axes]]: A tuple containing:
            - Figure: The matplotlib figure object containing both plots
            - List[Axes]: List of two axes objects for the image and depth map plots

    Raises:
        ValueError: If depth tensor dimensions don't match image dimensions
        TypeError: If depth is not a torch.Tensor or image is not a PIL Image

    Example:
        >>> from PIL import Image
        >>> import torch
        >>> # Load image and depth map
        >>> img = Image.open('image.png')
        >>> depth_map = torch.rand(img.size[::-1])  # Random depth for example
        >>> # Create the plot
        >>> fig, axes = plot_image_and_depthmap(img, depth_map)
        >>> plt.show()
    """
    # Input validation
    if not isinstance(image, Image):
        raise TypeError("image must be a PIL Image")
    if not isinstance(depth, torch.Tensor):
        raise TypeError("depth must be a torch.Tensor")

    if depth.shape[:2] != image.size[::-1]:
        raise ValueError("Depth dimensions must match image dimensions")

    # Create figure and axes
    fig: Figure
    ax: List[Axes]
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    # Plot the image
    ax[0].set_title(image_title)
    ax[0].imshow(image)
    ax[0].axis("off")  # Hide axes for cleaner visualization

    # Plot the depth map
    ax[1].set_title(depth_title)
    depth_plot = ax[1].imshow(depth.cpu(), cmap="jet_r")
    ax[1].axis("off")  # Hide axes for cleaner visualization

    # Add colorbar specifically for the depth plot
    plt.colorbar(depth_plot, ax=ax[1], label="Depth")

    # Adjust layout to prevent overlap
    plt.tight_layout()

    return fig, ax
