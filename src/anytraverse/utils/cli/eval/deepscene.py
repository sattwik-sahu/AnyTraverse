from time import perf_counter

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from rich.console import Console
from rich.progress import Progress
from torch.utils.data import DataLoader

from anytraverse.config.pipeline_002 import PipelineConfig
from anytraverse.config.utils import (
    CameraConfig,
    HeightScoringConfig,
    PlaneFittingConfig,
)
from anytraverse.utils.datasets.deepscene import DeepsceneDataset
from anytraverse.utils.helpers.plane_fit import PCAPlaneFitter
from anytraverse.utils.metrics.iou import iou_score
from anytraverse.utils.helpers.pooler import (
    WeightedMaxPooler,
    ProbabilisticPooler,
)
from anytraverse.utils.pipelines.base import Pipeline2
from anytraverse.utils.viz.pipeline import plot_anytraverse_output


def run_eval(show_viz: bool = True):
    # Define the console
    console = Console()

    # Create dataloader
    dataloader = DataLoader(
        dataset=DeepsceneDataset(
            images_root="/mnt/toshiba_hdd/datasets/deepscene/freiburg_forest_annotated/",
            masks_root="/mnt/toshiba_hdd/datasets/deepscene/freiburg_forest_annotated/",
            image_pattern="**/rgb/*.jpg",
            mask_pattern="**/GT_color/*.png",
        ),
        batch_size=1,
        shuffle=True,
    )

    # Define the pipeline
    fx, fy, cx, cy = 2813.643275, 2808.326079, 969.285772, 624.049972
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.cuda.is_available()
        else "cpu"
    )
    config = PipelineConfig(
        camera=CameraConfig(fx=fx, fy=fy, cx=cx, cy=cy),
        prompts=[("grass", 1.0), ("bush", -1.0), ("road", 1.0), ("water", -1.0)],
        device=device,
        height_scoring=HeightScoringConfig(alpha=(80, 30), z_thresh=(-0.1, 0.2)),
        plane_fitting=PlaneFittingConfig(
            fitter=PCAPlaneFitter(),
            trav_thresh=0.5,
        ),
        height_score=True,
        mask_pooler=WeightedMaxPooler(),
    )

    with console.status("Initializing pipeline..."):
        pipeline = Pipeline2(config=config)
    console.print("Pipeline initialized", style="bold green")

    # Get the batch
    batch = next(iter(dataloader))

    # Evaluate the pipeline
    threshold = 0.2
    mean_iou = 0.0
    plt.ion()

    # Create a figure and a 2x3 grid
    fig = plt.figure(figsize=(15, 8))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[5, 1])

    # Create the top three axes for the images
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # Merge the bottom row into a single axis for the text
    ax_text = fig.add_subplot(gs[1, :])

    # Combine the axes into a list to pass into the function
    ax = [ax1, ax2, ax3, ax_text]

    plt.show()

    with Progress() as progress:
        task = progress.add_task(
            f"Evaluating... Total = {len(dataloader)}", total=len(dataloader)
        )
        for i, batch in enumerate(dataloader):
            image_tensor, mask = batch["image"].squeeze(0), batch["mask"].squeeze(0)
            image = Image.fromarray(image_tensor.cpu().numpy())
            t0 = perf_counter()
            pred_mask = pipeline(image=image) > threshold
            t1 = perf_counter()

            score = iou_score(y_true=mask.cpu().numpy(), y_pred=pred_mask.cpu().numpy())
            mean_iou = (mean_iou * i + score) / (i + 1)
            progress.print(
                {
                    "n": i + 1,
                    "iou": float(np.round(score, 4)),
                    "mean_iou": float(np.round(mean_iou, 4)),
                    "time_ms": float(np.round((t1 - t0) * 1e3, 4)),
                }
            )

            progress.update(task, advance=1)

            if show_viz:
                for ax_ in ax:
                    ax_.clear()
                    ax_.set_axis_off()
                plot_anytraverse_output(
                    image=image,
                    mask_true=mask.cpu().numpy(),
                    mask_pred=pred_mask.cpu().numpy(),
                    params={
                        "$i$": i,
                        "IOU": np.round(score, 4),
                        "$IOU_{\\text{mean}}$": np.round(mean_iou, 4),
                        "$t_{\\text{inference}}$": np.round((t1 - t0) * 1e3, 4),
                    },
                    fig=fig,
                    ax=ax,
                )
                plt.tight_layout()
                plt.pause(0.01)

                # Check if window was closed
                if not plt.get_fignums():
                    break


if __name__ == "__main__":
    run_eval(show_viz=False)
