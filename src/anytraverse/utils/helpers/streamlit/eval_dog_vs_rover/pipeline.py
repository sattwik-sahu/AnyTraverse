import torch

from anytraverse.utils.pipelines.base import (
    Pipeline2,
    WeightedPrompt,
    PipelineOutput,
)
from anytraverse.config.pipeline_002 import PipelineConfig
from anytraverse.config.utils import (
    CameraConfig,
    HeightScoringConfig,
    PlaneFittingConfig,
)
from PIL import Image
from typing import List
from anytraverse.utils.helpers.pooler import WeightedMaxPooler
from anytraverse.utils.helpers.plane_fit import PCAPlaneFitter
import streamlit as st


fx = 1172.04
fy = 1175.24
cx = 716.08
cy = 554.43
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.cuda.is_available()
    else "cpu"
)


@st.cache_resource
def create_pipeline() -> Pipeline2:
    config = PipelineConfig(
        camera=CameraConfig(fx=fx, fy=fy, cx=cx, cy=cy),
        prompts=[],
        device=device,
        height_scoring=HeightScoringConfig(alpha=(75, 30), z_thresh=(-10.0, 0.1)),
        plane_fitting=PlaneFittingConfig(
            fitter=PCAPlaneFitter(),
            trav_thresh=0.3,
        ),
        height_score=True,
        mask_pooler=WeightedMaxPooler(),
    )
    pipeline = Pipeline2(config=config)
    return pipeline


def run_pipeline(
    pipeline: Pipeline2,
    prompts: List[WeightedPrompt],
    image: Image.Image,
    perform_height_scoring: bool = False,
) -> PipelineOutput:
    pipeline.prompts = prompts
    pipeline.perform_height_scoring = perform_height_scoring
    output = pipeline(image=image)
    return output
