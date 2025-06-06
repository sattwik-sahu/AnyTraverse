from time import sleep
from typing import Type

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from rich.console import Console

from anytraverse.config.utils import WeightedPrompt
from anytraverse.utils.cli.human_op.io import (
    get_prompts,
    get_weighted_prompt_from_string,
)
from anytraverse.utils.cli.human_op.models import (
    DriveStatus,
    HumanOperatorControllerState,
    ImageEmbeddings,
    Scene,
    SceneWeightedPrompt,
    Thresholds,
)
from anytraverse.utils.cli.human_op.prompt_store import ScenePromptStoreManager
from anytraverse.utils.helpers import DEVICE
from anytraverse.utils.helpers.human_op.uncertainty import (
    InvMaxProbaUncertaintyChecker,
    UncertaintyChecker,
)
from anytraverse.utils.helpers.mask_poolers import (
    MaskPooler,
    ProbabilisticPooler,
    WeightedMaxPooler,
)
from anytraverse.utils.metrics.roi import ROI_Checker
from anytraverse.utils.models import ImageEmbeddingModel
from anytraverse.utils.models.clip import CLIP
from anytraverse.utils.models.siglip import SigLIP
from anytraverse.utils.pipelines import create_pipeline
from anytraverse.utils.pipelines.base import Pipeline2 as Pipeline
from anytraverse.utils.pipelines.base import PipelineOutput

torch.set_default_device(device=DEVICE)


class AnyTraverseHOC_Context:
    _scene_prompt_store: ScenePromptStoreManager
    _image_embedding: ImageEmbeddingModel
    _drive_status: DriveStatus
    _pipeline: Pipeline
    _roi_checker: ROI_Checker
    _unc_checker: UncertaintyChecker
    _thresholds: Thresholds
    _curr_scene: Scene

    def __init__(
        self,
        image_embedding: ImageEmbeddings,
        roi_checker: ROI_Checker,
        unc_roi_thresh: float,
        seg_thresh: float,
        init_prompts: list[WeightedPrompt],
        ref_sim_thresh: float,
        trav_roi_thresh: float = 0.5,
        unc_checker: Type[UncertaintyChecker] = InvMaxProbaUncertaintyChecker,
        mask_pooler: Type[MaskPooler] = ProbabilisticPooler,
    ) -> None:
        self._thresholds = Thresholds(
            ref_sim=ref_sim_thresh,
            roi_unc=unc_roi_thresh,
            seg=seg_thresh,
            # trav_roi=trav_roi_thresh,
        )
        match image_embedding:
            case ImageEmbeddings.CLIP:
                self._image_embedding = CLIP(device=DEVICE)
            case ImageEmbeddings.SigLIP:
                self._image_embedding = SigLIP(device=DEVICE)
            case _:
                raise Exception("Expected value of `image_embedding` not to be `None`.")
        self._roi_checker = roi_checker
        self._unc_checker = unc_checker(roi=self._roi_checker)
        self._scene_prompt_store = ScenePromptStoreManager(
            image_embedding_model=self._image_embedding
        )
        self._drive_status = DriveStatus.OK
        self._pipeline = create_pipeline(
            init_prompts=init_prompts, mask_pooler=mask_pooler
        )
        self._pipeline.prompts = init_prompts

    @property
    def prompts(self) -> list[WeightedPrompt]:
        return self._pipeline.prompts

    def human_call_with_syntax(self, prompts_str: str) -> None:
        """
        Perform a human operator call with prompts given in a string
        following the syntax: "<prompt>: <weight>; <prompt>: <weight>; ...".

        Args:
            prompts_str (str): The string containing the weighted prompts.

        Raises:
            ValueError: If the string does not follow the expected syntax.
        """
        prompts: list[WeightedPrompt] = get_weighted_prompt_from_string(
            prompts_str=prompts_str
        )
        self.human_call(human_prompts=prompts)

    def human_call(self, human_prompts: list[WeightedPrompt]):
        """
        Performs a human operator call with `human_prompts` as the
        input from the human in the call. Updates the traversability
        preferences.

        Args:
            human_prompts (list[WeightedPrompt]): The weighted prompts
                input by the human operator
        """
        self._update_prompts(prompts=human_prompts)
        prompts: list[WeightedPrompt] = self._pipeline.prompts

        # Create a new ScenePrompt with the current frame and given prompts
        new_scene_prompt: SceneWeightedPrompt = SceneWeightedPrompt(
            scene=Scene(
                ref_frame=self._curr_scene["ref_frame"],
                ref_frame_embedding=self._curr_scene["ref_frame_embedding"],
            ),
            prompts=prompts,
        )
        # Add the new scene prompt to the memory
        self._scene_prompt_store.add_scene_prompt(scene_prompt=new_scene_prompt)

    def _update_prompts(self, prompts: list[WeightedPrompt]) -> None:
        # Create an updated version of the prompts
        curr_prompt_dict: dict[str, float] = dict[str, float](self._pipeline.prompts)
        prompts_dict: dict[str, float] = dict[str, float](prompts)
        updated_prompts_dict: dict[str, float] = {**curr_prompt_dict, **prompts_dict}
        updated_prompts: list[WeightedPrompt] = list(updated_prompts_dict.items())
        self._pipeline.prompts = updated_prompts

    def _get_masks(self, frame: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        anytraverse: PipelineOutput = self._pipeline(image=frame)
        return anytraverse.output, anytraverse.trav_masks

    def run_next(self, frame: Image.Image) -> HumanOperatorControllerState:
        # Create image embeddings for the frame
        frame_embedding: torch.Tensor = self._image_embedding(x=frame)
        # Create a new scene and set the current scene
        self._curr_scene = Scene(ref_frame=frame, ref_frame_embedding=frame_embedding)

        if len(self._scene_prompt_store._store) == 0:
            self._scene_prompt_store.add_scene_prompt(
                scene_prompt=SceneWeightedPrompt(
                    scene=self._curr_scene, prompts=self._pipeline.prompts
                )
            )

        # Check similarity with reference scene
        best_match_scene_prompt, best_match_sim = (
            self._scene_prompt_store.get_best_match(frame=frame)
        )
        # Check if viable best match
        if best_match_sim < self._thresholds["ref_sim"]:
            # Human operator should be called
            self._drive_status = DriveStatus.UNSEEN_SCENE
        elif self._drive_status is DriveStatus.UNSEEN_SCENE:
            # Was previously unseen scene, but now got corrected
            self._drive_status = DriveStatus.OK

        # Set current scene prompt to the best match
        self._update_prompts(prompts=best_match_scene_prompt["prompts"])

        # Pass the frame through the pipeline
        trav_mask, prompt_masks = self._get_masks(frame=frame)
        # print(prompt_masks.shape)

        # Check the uncertainty in the ROI
        unc_mask, roi_unc = self._unc_checker.roi_uncertainty(masks=prompt_masks)
        # Check if unknown object in ROI?
        if roi_unc > self._thresholds["roi_unc"]:
            # Unknown object found in ROI
            self._drive_status = DriveStatus.UNK_ROI_OBJ
        elif self._drive_status is DriveStatus.UNK_ROI_OBJ:
            # If an unknown object was in the ROI before
            # but it is known now, then we're good
            self._drive_status = DriveStatus.OK

        # Check the traversability ROI
        roi_trav: float = self._roi_checker.trav_area(mask=trav_mask)

        # print(
        #     {
        #         "thresholds": self._thresholds,
        #         "unc_roi": roi_unc,
        #         "ref_sim": best_match_sim,
        #     }
        # )

        return HumanOperatorControllerState(
            frame=frame,
            scene_prompt=SceneWeightedPrompt(
                scene=self._curr_scene, prompts=self._pipeline.prompts
            ),
            human_call=self._drive_status,
            trav_roi=roi_trav,
            unc_roi=roi_unc,
            unc_map=unc_mask,
            trav_map=trav_mask,
            prompt_attn_maps=prompt_masks,
        )


def create_anytraverse_hoc_context(
    init_prompts: list[WeightedPrompt],
    image_embedding: ImageEmbeddings = ImageEmbeddings.CLIP,
    roi_x_bounds: tuple[float, float] = (0.33, 0.67),
    roi_y_bounds: tuple[float, float] = (0.67, 1.00),
    seg_thresh: float = 0.25,
    ref_sim_thresh: float = 0.9,
    unc_roi_thresh: float = 0.5,
    unc_checker: Type[UncertaintyChecker] = InvMaxProbaUncertaintyChecker,
    mask_pooler: Type[MaskPooler] = WeightedMaxPooler,
) -> AnyTraverseHOC_Context:
    return AnyTraverseHOC_Context(
        image_embedding=image_embedding,
        roi_checker=ROI_Checker(
            x_bounds=roi_x_bounds,
            y_bounds=roi_y_bounds,
            device=DEVICE,
        ),
        seg_thresh=seg_thresh,
        ref_sim_thresh=ref_sim_thresh,
        unc_roi_thresh=unc_roi_thresh,
        init_prompts=init_prompts,
        unc_checker=unc_checker,
        mask_pooler=mask_pooler,
    )


def main():
    console = Console()
    plt.ion()
    fig, ax = plt.subplots(1, 2, figsize=(24, 16))
    fig.show()

    FPS = 30

    hoc_ctx = AnyTraverseHOC_Context(
        image_embedding=ImageEmbeddings.CLIP,
        ref_sim_thresh=0.9,
        roi_checker=ROI_Checker(
            x_bounds=(0.33, 0.67), y_bounds=(0.67, 1.00), device=DEVICE
        ),
        seg_thresh=0.25,
        trav_roi_thresh=0.5,
        unc_checker=InvMaxProbaUncertaintyChecker,
        unc_roi_thresh=0.5,
        init_prompts=[("grass", 1)],
    )
    hoc_ctx._pipeline._mask_pooler = ProbabilisticPooler()
    cap = cv2.VideoCapture(
        filename="/mnt/toshiba_hdd/datasets/iiserb/anytraverse/2024-12-10__hound_hillside/video_012.avi"
    )
    console.log(hoc_ctx._pipeline.prompts)
    while cap.isOpened():
        ret, frame_ = cap.read()
        if not ret:
            print("===== END OF VIDEO =====")
            break
        rgb_frame: np.ndarray = cv2.cvtColor(frame_, cv2.COLOR_BGR2RGB)
        frame: Image.Image = Image.fromarray(rgb_frame)
        console.log(f"Frame size: {frame.size}")

        state = hoc_ctx.run_next(frame=frame)
        console.log(
            f"hoc={state.human_call}, trav_roi={state.trav_roi}, unc_roi={state.unc_roi}"
        )
        console.log(f"Prompt masks shape: {state.prompt_attn_maps.shape}")

        ax[0].clear()
        ax[1].clear()
        ax[0].imshow(frame)
        ax[0].imshow(state.unc_map.cpu().numpy(), alpha=0.4, cmap="plasma")
        ax[0].set_axis_off()
        ax[0].set_title("Uncertainty Map")

        ax[1].imshow(frame)
        ax[1].imshow(state.trav_map.cpu().numpy(), alpha=0.4, cmap="plasma")
        ax[1].set_axis_off()
        ax[1].set_title("Traversability Map")

        console.log(f"prompts={dict(hoc_ctx._pipeline.prompts)}")

        if state.human_call is DriveStatus.OK:
            console.log("No problem desu!", style="bold light_green")
        else:
            console.log("Human Operator call required", style="red")
            match state.human_call:
                case DriveStatus.UNK_ROI_OBJ:
                    console.log(
                        "Never seen this shit in my entire `episode`", style="yellow"
                    )
                case DriveStatus.UNSEEN_SCENE:
                    console.log(
                        "Never been here in my entire `episode`", style="magenta"
                    )
            console.log("Enter new prompts mommy", style="dim")
            hoc_ctx.human_call(human_prompts=get_prompts(console=console))
        plt.pause(1 / FPS)


if __name__ == "__main__":
    main()
