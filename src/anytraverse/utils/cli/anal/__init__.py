from pathlib import Path
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
from anytraverse.utils.data.rugd import RUGD_Dataset
from anytraverse.utils.helpers.plane_fit import PCAPlaneFitter
from anytraverse.utils.metrics.iou import iou_score
from anytraverse.utils.helpers.pooler import (
    WeightedMaxPooler,
    ProbabilisticPooler,
)
from anytraverse.utils.pipelines.base import Pipeline2

from anytraverse.utils.cli.anal.viz import run


def run_anal():
    # Define the console
    console = Console()

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
        prompts=[("grass", 0.9), ("bush", -0.8), ("dirt path", 1.0)],
        device=device,
        height_scoring=HeightScoringConfig(alpha=30, z_thresh=0.1),
        plane_fitting=PlaneFittingConfig(
            fitter=PCAPlaneFitter(),
            trav_thresh=0.1,
        ),
        height_score=True,
        # mask_pooler=ProbabilisticPooler(),
        mask_pooler=WeightedMaxPooler(),
    )

    with console.status("Initializing pipeline..."):
        pipeline = Pipeline2(config=config)
    console.print("Pipeline initialized", style="bold green")

    # Evaluate the pipeline
    threshold = 0.4

    run(
        pipeline=pipeline,
        video_path=Path(
            "/mnt/toshiba_hdd/datasets/iiserb/anytraverse/2024-11-18__dog_hillside-iiserb-001.avi"
        ),
        output_path=Path(
            "/mnt/toshiba_hdd/datasets/iiserb/anytraverse/analysis/2024-11-18__iiserb-hillside-001"
        ),
        threshold=threshold,
        nth_frame=10,
        color=(255, 80, 150),
        overlay_alpha=0.25,
    )


if __name__ == "__main__":
    run_anal()
