from pathlib import Path
from time import perf_counter_ns
from typing import List

import torch
import typer
from matplotlib import pyplot as plt
from PIL.Image import Image
from PIL.Image import open as open_image
from rich.console import Console
from typing_extensions import Annotated

from anytraverse.utils.models.clipseg.model import CLIPSeg


def clipseg(
    image_path: Annotated[Path, typer.Argument(help="The image path")],
    prompts: Annotated[
        List[str], typer.Option(prompt_required=True, help="List of prompt strings")
    ],
):
    console = Console()

    with console.status(f"Opening image {image_path.as_posix()}"):
        image: Image = open_image(image_path)
    console.log(f"Loaded image of size {image.size[::-1]}")

    # Initialize model
    with console.status("Loading CLIPSeg model..."):
        clipseg = CLIPSeg(device="cuda" if torch.cuda.is_available() else "cpu")
    console.log(
        f"Model loaded. Memory: {clipseg._model.get_memory_footprint() / (1024 ** 2) :.3f} Mb"
    )

    # Run the inference
    t0 = perf_counter_ns()
    with console.status("Running model on inputs..."):
        output = clipseg(prompts=prompts, image=image)
    console.log(f"Completed inference in {(perf_counter_ns() - t0) / 1e9 :.3f} s")

    # Load output to CPU for plotting results
    if clipseg._device.type != "cpu":
        output = output.cpu()

    # Plot the results
    fig, ax = plt.subplots(nrows=1, ncols=len(prompts) + 1)

    # Plot input image
    ax[0].imshow(image)
    ax[0].set_title("Input image")
    ax[0].axis("off")

    # Plot the output masks (overlayed)
    for ax_, prompt, output_mask in zip(ax[1:], prompts, output):
        ax_.imshow(image)
        ax_.imshow(output_mask[0], alpha=0.5, cmap="jet")
        ax_.set_title(prompt)
        ax_.axis("off")

    plt.show()
