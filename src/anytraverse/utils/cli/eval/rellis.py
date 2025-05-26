from matplotlib.gridspec import GridSpec
import torch
import numpy as np

from anytraverse.utils.pipelines.base import Pipeline2
from anytraverse.config.pipeline_002 import PipelineConfig
from anytraverse.config.utils import (
    CameraConfig,
    HeightScoringConfig,
    PlaneFittingConfig,
)
from anytraverse.utils.helpers.pooler import (
    WeightedMaxPooler,
    ProbabilisticPooler,
)
from anytraverse.utils.helpers.plane_fit import PCAPlaneFitter
from anytraverse.utils.datasets.rellis import RellisDataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from PIL import Image
from anytraverse.utils.metrics.iou import iou_score
from time import perf_counter
from rich.console import Console
from rich.progress import Progress
from anytraverse.utils.viz.pipeline import plot_anytraverse_output
from anytraverse.utils.metrics.live_mean_std import LiveMeanStd


def run_eval(show_viz: bool = True):
    # Define the console
    console = Console()

    # Define the paths
    images_root = "/mnt/toshiba_hdd/datasets/rellis-3d/Rellis-3D-images/"
    masks_root = "/mnt/toshiba_hdd/datasets/rellis-3d/Rellis-3D-masks/"
    paths_list_path = "/mnt/toshiba_hdd/datasets/rellis-3d/splits/train.lst"

    # Create the dataloader
    rellis_dataloader = DataLoader(
        dataset=RellisDataset(
            images_root=images_root,
            path_list_path=paths_list_path,
            masks_root=masks_root,
        ),
        batch_size=1,
        shuffle=True,
    )

    # Define the pipeline
    fx, fy, cx, cy = 2813.643275, 2808.326079, 969.285772, 624.049972
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.cuda.is_available()
        else "cpu"
    )
    config = PipelineConfig(
        camera=CameraConfig(fx=fx, fy=fy, cx=cx, cy=cy),
        prompts=[
            ("grass", 1.0),
            ("bush", -1.0),
            ("dirt", 1.0),
        ],
        device=device,
        height_scoring=HeightScoringConfig(alpha=(75, 30), z_thresh=(-0.1, 0.1)),
        plane_fitting=PlaneFittingConfig(
            fitter=PCAPlaneFitter(),
            trav_thresh=0.3,
        ),
        height_score=True,
        # mask_pooler=ProbabilisticPooler(),
        mask_pooler=WeightedMaxPooler(),
    )

    with console.status("Initializing pipeline..."):
        pipeline = Pipeline2(config=config)
    console.print("Pipeline initialized", style="bold green")

    # Evaluate the pipeline
    threshold = 0.5
    mean_iou = LiveMeanStd()
    t_inf = LiveMeanStd()
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
        task = progress.add_task("Evaluating...", total=len(rellis_dataloader))
        for i, batch in enumerate(rellis_dataloader):
            image_tensor, mask = batch["image"].squeeze(0), batch["mask"].squeeze(0)
            image = Image.fromarray(image_tensor.cpu().numpy())
            t0 = perf_counter()
            pred_mask = pipeline(image=image).output > threshold
            t1 = perf_counter()
            t_inf.update(value=(t1 - t0) * 1e3)

            iou = iou_score(y_true=mask.cpu().numpy(), y_pred=pred_mask.cpu().numpy())
            mean_iou.update(value=iou)
            progress.print(
                {
                    "n": i + 1,
                    "iou": float(np.round(iou, 4)),
                    "mean_iou": mean_iou.to_plus_minus(n=4),
                    "time_ms": round(t_inf.latest, 4),
                    "mean_time_ms": round(t_inf.mean, 4),
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
                        "Index": i,
                        "IOU": np.round(iou, 4),
                        "Mean IOU": mean_iou.to_plus_minus(n=4),
                        "Time (ms)": np.round(t_inf.latest, 4),
                        "Mean Time (ms)": np.round(t_inf.mean),
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
    run_eval()
