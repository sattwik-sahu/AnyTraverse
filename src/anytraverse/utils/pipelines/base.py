from abc import ABC, abstractmethod
import torch
from PIL.Image import Image
from typing_extensions import override

from anytraverse.config.pipeline_002 import PipelineConfig, WeightedPrompt
from anytraverse.utils.models.clipseg.model import CLIPSeg

from anytraverse.utils.helpers.mask_poolers import MaskPooler, WeightedMaxPooler
from anytraverse.utils.helpers import DEVICE

from typing import List, NamedTuple, Type


class PipelineOutput(NamedTuple):
    trav_masks: torch.Tensor
    pooled_mask: torch.Tensor
    # height_scores: HeightScoringOutput | None
    output: torch.Tensor


class AnyTraversePipeline(ABC):
    _name: str

    def __init__(self, name: str = "Pipeline", *args, **kwargs) -> None:
        self._name = name

    @abstractmethod
    def __call__(self, image: Image) -> PipelineOutput:
        pass


class Pipeline2(AnyTraversePipeline):
    """
    Pipeline for traversability segmentation using CLIPSeg.
    This pipeline uses CLIPSeg to generate traversability masks based on provided prompts,
    pools these masks, and optionally applies height scoring to refine the output.

    Attributes:
        _clipseg (CLIPSeg): The model for generating traversability masks.
        _config (PipelineConfig): Configuration for the pipeline.
        _mask_pooler (CLIPSegMaskPooler): Pooler for combining multiple masks.
        _height_scoring_pipeline (HeightScoringPipeline): Pipeline for height scoring.
        _perform_height_scoring (bool): Flag to determine if height scoring should be performed.
        _analysis (PipelineOutput): Output of the pipeline containing masks and scores.

    TODO Generalize CLIPSeg to support any model that takes prompts and images and outputs attention maps.
    """

    _clipseg: CLIPSeg
    _config: PipelineConfig
    _mask_pooler: MaskPooler
    # _height_scoring_pipeline: HeightScoringPipeline
    _perform_height_scoring: bool

    _analysis: PipelineOutput

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(name="Pipeline_002")
        self._config = config
        self._clipseg = CLIPSeg(
            model_name="mcmonkey/clipseg-rd64-refined-fp16", device=self._config.device
        )
        # We don't need height scoring anymore
        self._perform_height_scoring = False

        # self._perform_height_scoring = (
        #     config.plane_fitting is not None
        #     and config.height_scoring is not None
        #     and self._config.height_score
        # )
        # if self._perform_height_scoring:
        #     self._height_scoring_pipeline = HeightScoringPipeline(
        #         plane_fitter=self._config.plane_fitting.fitter,  # type: ignore
        #         alpha=self._config.height_scoring.alpha,  # type: ignore
        #         z_thresh=self._config.height_scoring.z_thresh,  # type: ignore
        #         camera_config=self._config.camera,
        #         device=self._config.device,
        #     )

    @property
    def prompts(self) -> List[WeightedPrompt]:
        return self._config.prompts

    @prompts.setter
    def prompts(self, prompts_: List[WeightedPrompt]) -> None:
        self._config.prompts = prompts_

    # @property
    # def perform_height_scoring(self) -> bool:
    #     return self._config.height_score

    # @perform_height_scoring.setter
    # def perform_height_scoring(self, perform_height_scoring_: bool) -> None:
    #     self._config.height_score = perform_height_scoring_

    @override
    def __call__(self, image: Image) -> PipelineOutput:
        trav_masks: torch.Tensor = self._clipseg(
            image=image, prompts=[p[0] for p in self._config.prompts]
        )  # Dimensions: (num_prompts, 1, H, W)

        pooled_trav_mask: torch.Tensor = self._config.mask_pooler.pool(
            masks=trav_masks,
            weights=[p[1] for p in self._config.prompts],
            device=self._config.device,
        )  # Dimensions: (H, W)

        # height_scores: HeightScoringOutput | None = None
        # if self._config.height_score:
        #     height_scores = self._height_scoring_pipeline(
        #         image=image,
        #         plane_fit_mask=pooled_trav_mask
        #         > self._config.plane_fitting.trav_thresh,  # type: ignore
        #     )  # Dimensions: (H, W)

        #     # convert height scores to float32
        #     z_scores = height_scores.scores.type(torch.float16)

        #     # Combine the two scores
        #     final_output = pooled_trav_mask.to(
        #         device=self._config.device
        #     ) * z_scores.to(self._config.device)
        # else:

        # The final output is just the pooled traversability mask
        final_output = pooled_trav_mask  # Dimensions: (H, W)

        return PipelineOutput(
            trav_masks=trav_masks,
            pooled_mask=pooled_trav_mask,
            # height_scores=height_scores,
            output=final_output,
        )


def create_pipeline(
    init_prompts: list[WeightedPrompt],
    mask_pooler: Type[MaskPooler] = WeightedMaxPooler,
) -> Pipeline2:
    """
    Factory function to create an instance of Pipeline2 with default configuration.

    Returns:
        Pipeline2: An instance of the Pipeline2 class with default configuration.
    """
    config = PipelineConfig(
        prompts=init_prompts,
        mask_pooler=mask_pooler,
        device=DEVICE,
    )

    return Pipeline2(config=config)
