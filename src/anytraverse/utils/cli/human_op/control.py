import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from PIL import Image
from rich.console import Console
from rich.panel import Panel
from torchvision.transforms import ToPILImage

from anytraverse.config.utils import WeightedPrompt
from anytraverse.utils.cli.human_op.io import (
    dict_to_table,
    get_history_pickle_path,
    get_prompts,
    op_call_req_confirm,
    save_log,
)
from anytraverse.utils.cli.human_op.models import (
    DatasetVideo,
    DriveStatus,
    HumanOperatorCallLogModel,
    LoopbackLogModel,
    LoopbackLogsModel,
    Thresholds,
    Scene,
    SceneWeightedPrompt,
)
from anytraverse.utils.cli.human_op.prompt_store import ScenePromptStoreManager
from anytraverse.utils.cli.human_op.video import get_video_path
from anytraverse.utils.helpers.human_op.prompts import Prompts, update_prompts
from anytraverse.utils.helpers.human_op.uncertainty import (
    ProbabilisticUncertaintyChecker,
    UncertaintyChecker,
)
from anytraverse.utils.metrics.roi import ROI_Checker
from anytraverse.utils.models import ImageEmbeddingModel
from anytraverse.utils.pipelines.base import Pipeline2 as Pipeline
from anytraverse.utils.pipelines.base import PipelineOutput
from anytraverse.utils.pipelines.base import create_pipeline
from anytraverse.utils.viz.human_op import HumanOperatorUI_Axes, get_human_op_ui_axes
from anytraverse.utils.viz.roi import plot_image_seg_roi


class HumanOperatorController:
    _video: DatasetVideo
    _capture: cv2.VideoCapture
    _prompt_store: ScenePromptStoreManager
    _inx: int
    _frame: Image.Image
    _curr_scene_prompt: SceneWeightedPrompt
    _to_pil: ToPILImage
    _pipeline: Pipeline
    _console: Console
    _roi: ROI_Checker
    _fig: Figure
    _ax: HumanOperatorUI_Axes
    _thresholds: Thresholds
    _n_frames_skip: int
    _drive_status: DriveStatus
    _image_embedding_model: ImageEmbeddingModel
    _uncertainty_checker: UncertaintyChecker
    _hoc_progression: Dict[DriveStatus, List[bool]]
    _scene_similarities: List[float]

    def __init__(
        self,
        video: DatasetVideo,
        video_path: Path,
        console: Console,
        image_embedding_model: ImageEmbeddingModel,
        roi_unc_thresh: float = 0.5,
        ref_sim_thresh: float = 0.9,
        seg_thresh: float = 0.25,
        n_frames_skip: int = 0,
    ) -> None:
        self._video = video
        self._console = console
        self._n_frames_skip = n_frames_skip
        self._capture = cv2.VideoCapture(video_path.as_posix())
        self._inx = 0
        self._to_pil = ToPILImage()
        self._frame = self._read_next_frame()  # type: ignore
        self._roi = ROI_Checker(
            x_bounds=(0.33, 0.67), y_bounds=(0.67, 1.00), device=torch.device("cuda")
        )

        self._hoc_progression = {
            DriveStatus.UNK_ROI_OBJ: [],
            DriveStatus.UNSEEN_SCENE: [],
        }
        self._scene_similarities = []

        # TODO Make this generalizable to different types of checkers
        self._uncertainty_checker = ProbabilisticUncertaintyChecker(roi=self._roi)

        self._thresholds = Thresholds(
            ref_sim=ref_sim_thresh, roi_unc=roi_unc_thresh, seg=seg_thresh
        )

        with console.status("Initializing pipeline..."):
            self._pipeline = create_pipeline()
        console.print(
            "Initialized [bold light_green]AnyTraverse[/] pipeline successfully!"
        )

        self._image_embedding_model = image_embedding_model
        self._prompt_store = ScenePromptStoreManager(
            image_embedding_model=self._image_embedding_model
        )

        # Create the figure are show on screen
        self._setup_plot()

        # Input initial prompts from user
        self._setup_initial_prompts()

        # Set drive status OK
        self._drive_status = DriveStatus.OK

    @property
    def prompts_hist_store(self) -> List[SceneWeightedPrompt]:
        return self._prompt_store._store

    def help_me_mommy(self) -> Prompts:
        """
        The actual human operator gets called here, and the
        prompts and weights get updates using the inputs from the human operator.

        Returns:
            Prompts: The updated prompts after the human inputs the changes.
        """
        prompts_user: Prompts = get_prompts()
        if prompts_user:
            return update_prompts(
                prompts=self._pipeline.prompts, delta_prompts=prompts_user
            )
        else:
            return self._pipeline.prompts

    def _read_next_frame(self) -> Image.Image | None:
        # Read next frame after skipping some frames
        for _ in range(self._n_frames_skip if self._inx > 0 else 0):
            if self._read_single_frame() is None:
                return None
        frame = self._read_single_frame()

        if frame is not None:
            return frame
        else:
            return None

    def _show_image_seg_roi_plot(
        self, trav_mask: torch.Tensor, unc_mask: torch.Tensor
    ) -> None:
        self._reset_plot()

        # self._ax[0].set_title("Current Scene Reference Frame")
        # self._ax[1].set_title(f"Current Frame (index: {self._inx + 1})")

        self._ax.ref_frame.imshow(self._curr_scene_prompt["scene"]["ref_frame"])
        self._ax.curr_frame.imshow(self._frame)

        # Traversability Map
        plot_image_seg_roi(
            ax=self._ax.trav_map,
            image=self._frame,
            mask=trav_mask,
            threshold=self._thresholds["seg"],
            roi=self._roi,
            msg="Traversability",
            color=(95, 214, 127),
        )

        # Uncertainty Map
        plot_image_seg_roi(
            ax=self._ax.unc_map,
            image=self._frame,
            mask=unc_mask,
            threshold=self._thresholds["roi_unc"],
            roi=self._roi,
            msg="Uncertainty",
            color=(202, 5, 77),
        )

        # HOC Progression
        self._ax.hoc.stairs(
            np.cumsum(self._hoc_progression[DriveStatus.UNK_ROI_OBJ]),
            label="Unknown Object",
        )
        self._ax.hoc.stairs(
            np.cumsum(self._hoc_progression[DriveStatus.UNSEEN_SCENE]),
            label="Unseen Scene",
        )
        self._ax.hoc.stairs(
            np.logical_or(
                self._hoc_progression[DriveStatus.UNK_ROI_OBJ],
                self._hoc_progression[DriveStatus.UNSEEN_SCENE],
            ).cumsum(),
            linestyle="dashdot",
            label="Combined HOC",
        )

        self._ax.hoc.legend()

        # Plot ref scene sim dist
        if len(self._scene_similarities) > 3:
            g_sc = sns.histplot(data=np.array(self._scene_similarities), kde=True)
            g_sc.axvline(self._thresholds["ref_sim"])

    def _setup_plot(self) -> None:
        plt.ion()
        self._fig, self._ax = get_human_op_ui_axes()
        self._fig.show()

    def _reset_plot(self) -> None:
        self._ax.reset()

    def _setup_initial_prompts(self) -> None:
        # Show frame
        self._reset_plot()
        self._ax.curr_frame.imshow(self._frame)
        # self._ax[0].set_title("Frame")

        # Get prompts
        prompts: List[WeightedPrompt] = get_prompts()
        self._console.log(f"Got prompts: {prompts}")

        # Save prompts to prompts store
        scene_prompt = SceneWeightedPrompt(
            scene=Scene(
                ref_frame=self._frame,
                ref_frame_embedding=self._image_embedding_model(x=self._frame),
            ),
            prompts=prompts,
        )
        self._prompt_store.add_scene_prompt(scene_prompt=scene_prompt)
        self._curr_scene_prompt = scene_prompt
        self._pipeline.prompts = prompts

    def _show_info_table(self, kv: Dict[str, Any]) -> None:
        # Convert to Dict[str, str] before passing in
        table = dict_to_table(kv={k: str(v) for k, v in kv.items()})
        self._console.print(table)

    def _anytraverse_on_current_frame(self) -> PipelineOutput:
        # Set the prompts
        self._pipeline.prompts = self._curr_scene_prompt["prompts"]
        # Run the pipeline
        return self._pipeline(image=self._frame)

    def _get_ref_frame_sim_score(self) -> float:
        """
        Gets the similarity score of the current frame with the scene
        reference frame.
        """
        return float(
            torch.cosine_similarity(
                x1=self._image_embedding_model(
                    x=self._curr_scene_prompt["scene"]["ref_frame"]
                ),
                x2=self._image_embedding_model(x=self._frame),
            ).item()
        )

    def _read_single_frame(self) -> Image.Image | None:
        ret, frame = self._capture.read()
        if ret:
            rgb_frame: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb_frame)
        else:
            return None

    def _loopback(self) -> None:
        # Reset video capture
        self._capture = cv2.VideoCapture(get_video_path(self._video).as_posix())

        # Reset frame counter
        self._inx = 0

        # Log of current video
        video_log: LoopbackLogModel = LoopbackLogModel(
            video=self._video, roi_calls=[], scene_change_calls=[]
        )

        # Loop over all frames now
        with self._console.status("🔄 Looping back") as status:
            while self._capture.isOpened():
                # Read a frame
                frame: Image.Image | None = self._read_single_frame()
                if frame is None:
                    break

                # Get best match for frame from prompts store
                best_match_scene_prompt, best_match_ref_sim = (
                    self._prompt_store.get_best_match(frame=frame)
                )

                # Update current scene prompt information
                self._curr_scene_prompt = best_match_scene_prompt
                self._pipeline.prompts = self._curr_scene_prompt["prompts"]

                # Run the pipeline, get the traversability of ROI
                attn_map: torch.Tensor = self._pipeline(image=frame).output
                seg_mask: torch.Tensor = attn_map > self._thresholds["seg"]
                trav_roi: float = self._roi.trav_area(mask=seg_mask)

                # Check for human operator call conditions
                # Scene change?
                if best_match_ref_sim < self._thresholds["ref_sim"]:
                    video_log.scene_change_calls.append(self._inx)
                # # Low ROI?
                # if trav_roi < self._thresholds["roi"]:
                #     video_log.roi_calls.append(self._inx)

                # Update frame counter
                self._inx += 1

                # Update status
                status.update(f"🥸 Processed {self._inx} frames")

        print()
        self._console.print("✨ Loopback complete!", style="cyan")
        self._console.print(
            f"Scene change human calls = {len(video_log.scene_change_calls)}"
        )
        self._console.print(f"Low ROI human calls = {len(video_log.roi_calls)}")

        # Save log for this video
        with self._console.status("Saving loopback logs..."):
            with open("data/logs/loopback.json", "r") as f:
                logs: LoopbackLogsModel = LoopbackLogsModel.model_validate_json(
                    f.read()
                )
            logs.logs.append(video_log)
            with open("data/logs/loopback.json", "w") as f:
                f.write(logs.model_dump_json())
        self._console.print("😺 Saved loopback logs", style="light_green")

    def __call__(self) -> None:
        while self._capture.isOpened():
            # Should the data be logged?
            log_required: bool = True

            # Was hist used?
            hist_used_succ: bool = False

            # Get the next frame if not on first frame and drive status is OK
            if self._inx > 0:
                next_frame: Image.Image | None = self._read_next_frame()
                if next_frame is not None:
                    # Video is not over
                    self._frame = next_frame
                else:
                    # Video is over
                    break

            # Show frame index
            self._console.print(
                f"### FRAME {self._inx + 1} ###", justify="center", style="bold cyan"
            )

            # Get AnyTraverse output and ROI trav
            anytraverse_output: PipelineOutput = self._anytraverse_on_current_frame()

            # Get the masks
            trav_masks: torch.Tensor = anytraverse_output.trav_masks
            pooled_trav_mask: torch.Tensor = anytraverse_output.output

            thresh_mask: torch.Tensor = pooled_trav_mask > self._thresholds["seg"]

            # ROI calculations
            trav_roi: float = self._roi.trav_area(mask=thresh_mask)
            unc_mask, unc_roi = self._uncertainty_checker.roi_uncertainty(
                masks=trav_masks
            )

            # Calculate current frame vs. scene ref frame similarity
            ref_sim_score: float = self._get_ref_frame_sim_score()
            self._scene_similarities.append(ref_sim_score)

            with self._console.status(f"Plotting frame {self._inx}"):
                self._show_image_seg_roi_plot(
                    trav_mask=pooled_trav_mask,
                    unc_mask=self._uncertainty_checker._get_uncertainty_mask(
                        masks=trav_masks
                    ),
                )

            # Ask human, "Is an operator call required at this frame?"
            # human_op_call_required: bool = op_call_req_confirm(console=self._console)
            human_op_call_required: bool = False

            # Check ROI
            # if trav_roi < self._thresholds["roi"]:
            #     # Bad ROI
            #     self._drive_status = DriveStatus.BAD_ROI

            # How uncertain is the ROI?
            if unc_roi > self._thresholds["roi_unc"]:
                self._drive_status = DriveStatus.UNK_ROI_OBJ
            elif self._drive_status is not DriveStatus.OK:
                # Frame was NOT OK, but has become OK after human operator helps
                self._drive_status = DriveStatus.OK

                # Since this prompt works for the new scene, save it as a valid
                # scene prompt pair in the store (history)
                self._prompt_store.add_scene_prompt(self._curr_scene_prompt)

                # Since this is a resolution of a human operator call,
                # data should not be logged
                log_required = False
            else:
                # Check sim with current scene ref frame
                if ref_sim_score < self._thresholds["ref_sim"]:
                    # Get best match from history
                    hist_best_scene_prompt, hist_best_ref_sim = (
                        self._prompt_store.get_best_match(frame=self._frame)
                    )

                    # Check if best match fulfils criteria
                    if hist_best_ref_sim < self._thresholds["ref_sim"]:
                        # No satisfactory match found, so this is an unseen scene
                        self._drive_status = DriveStatus.UNSEEN_SCENE
                    else:
                        # Best match is actually a pretty similar frame to the current one
                        hist_used_succ = True
                        match_scene_prompt: SceneWeightedPrompt = (
                            hist_best_scene_prompt.copy()
                        )
                        match_scene_prompt["prompts"] = update_prompts(
                            prompts=self._curr_scene_prompt["prompts"],
                            delta_prompts=match_scene_prompt["prompts"],
                        )
                        self._curr_scene_prompt = match_scene_prompt

            drive_status_msg: str = "[bold lightgreen]OK[/]"
            if self._drive_status is DriveStatus.UNK_ROI_OBJ:
                drive_status_msg = "[red]Unknown object detected![/]"
            elif self._drive_status is DriveStatus.UNSEEN_SCENE:
                drive_status_msg = "[red]Unseen env detected![/]"

            self._show_info_table(
                kv={
                    "frame#": self._inx + 1,
                    "ref_sim_score": ref_sim_score
                    if ref_sim_score > -np.inf
                    else "[dim]NA[/]",
                    "trav_roi": f"{trav_roi * 100:.2f}%",
                    "roi_uncertainty": f"{unc_roi * 100:.2f}%",
                    "drive_status": drive_status_msg,
                }
            )
            self._console.print_json(
                json.dumps(dict(self._curr_scene_prompt["prompts"])),
            )

            if self._drive_status is not DriveStatus.OK:
                self._curr_scene_prompt = SceneWeightedPrompt(
                    scene=Scene(
                        ref_frame=self._frame,
                        ref_frame_embedding=self._image_embedding_model(self._frame),
                    ),
                    prompts=self.help_me_mommy(),
                )

            if log_required:
                # Create the data structure to log
                log = HumanOperatorCallLogModel(
                    frame_inx=self._inx,
                    human_op_call_req=human_op_call_required,
                    human_op_call_type=self._drive_status,
                    human_op_called=self._drive_status is not DriveStatus.OK,
                    ref_sim_score=ref_sim_score,
                    thresh=self._thresholds,
                    trav_roi=trav_roi,
                    unc_roi=unc_roi,
                    video=self._video,
                    prompts=self._pipeline.prompts,
                    hist_used_succ=hist_used_succ,
                )
                save_log(log=log)

                self._hoc_progression[DriveStatus.UNK_ROI_OBJ].append(
                    self._drive_status is DriveStatus.UNK_ROI_OBJ
                )
                self._hoc_progression[DriveStatus.UNSEEN_SCENE].append(
                    self._drive_status is DriveStatus.UNSEEN_SCENE
                )

            print("=" * 40, end="\n\n\n")
            self._inx += self._n_frames_skip
            plt.pause(0.003)

        self._capture.release()
        print()
        self._console.print("/// THE END ///", justify="center", style="bold magenta")

        print()
