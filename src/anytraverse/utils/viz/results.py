import time

import torch
from matplotlib import pyplot as plt
from PIL import Image

from anytraverse.config.pipeline_002 import PipelineConfig
from anytraverse.config.utils import (
    CameraConfig,
    HeightScoringConfig,
    PlaneFittingConfig,
)
from anytraverse.utils.helpers.plane_fit import PCAPlaneFitter
from anytraverse.utils.helpers.pooler import WeightedSumPooler
from anytraverse.utils.pipelines.base import Pipeline2
from anytraverse.utils.viz.clipseg import plot_clipseg_output as plot_output
from rich.console import Console


def main():
    console = Console()

    config = PipelineConfig(
        camera=CameraConfig(
            fx=2813.643275, fy=2808.326079, cx=969.285772, cy=624.049972
        ),
        prompts=[("grass", 1.0), ("bushes", -0.8), ("puddle", -0.8)],
        mask_pooler=WeightedSumPooler(),
        device="cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.cuda.is_available()
        else "cpu",
        # NOTE Include these parameters in config for height scoring 😼
        height_scoring=HeightScoringConfig(alpha=5, z_thresh=0.3),
        plane_fitting=PlaneFittingConfig(
            fitter=PCAPlaneFitter(),
            trav_thresh=0.5,
        ),
    )

    with console.status("Creating pipeline..."):
        t0 = time.perf_counter_ns()
        pipeline = Pipeline2(config)
        delta_t = time.perf_counter_ns() - t0
    console.log(f"Pipeline created in {delta_t * 1e-6:.4f} ms")

    image = Image.open(input("Enter the path to the image: "))

    with console.status("Running pipeline..."):
        t0 = time.perf_counter_ns()
        output = pipeline(image=image)  # Dimensions: (H, W)
        delta_t = time.perf_counter_ns() - t0
    console.log(f"Pipeline took: {delta_t * 1e-6:.4f} ms")

    plot_output(image=image, prompts=["Output"], masks=output.unsqueeze(0).unsqueeze(0))
    plt.show()


if __name__ == "__main__":
    main()
