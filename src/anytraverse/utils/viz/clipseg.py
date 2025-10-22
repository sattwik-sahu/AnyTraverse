import torch
from matplotlib import pyplot as plt
from PIL.Image import Image
from typing import Tuple, List
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def plot_clipseg_output(
    image: Image, prompts: List[str], masks: torch.Tensor, alpha: float = 0.5
) -> Tuple[Figure, List[Axes]]:
    fig: Figure
    ax: List[Axes]

    fig, ax = plt.subplots(nrows=1, ncols=len(prompts) + 1, figsize=(24, 16))

    plt.tight_layout()

    ax[0].set_title("Image")
    ax[0].imshow(image)

    # Plot the masks overlayed on the image
    masks_arr = masks.squeeze(1).cpu().numpy()
    for ax_, mask, prompt in zip(ax[1:], masks_arr, prompts):
        ax_.set_title(prompt)
        ax_.imshow(image)
        ax_.imshow(mask, cmap="jet", alpha=alpha)

    return fig, ax
