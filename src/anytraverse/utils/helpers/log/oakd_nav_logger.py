import torch
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime as dt
from pydantic import BaseModel
from anytraverse.config.pipeline_002 import WeightedPrompt
from anytraverse.utils.cli.human_op.models import (
    DriveStatus,
    HumanOperatorControllerState as AnyTraverseState,
)
from numpy import typing as npt
from anytraverse.utils.helpers.mask import torch_tensor_to_cv_image


type Cv2Image = npt.NDArray[np.uint8]


class AnyTraverseLogModel(BaseModel):
    roi_trav: float
    roi_unc: float
    prompts: list[WeightedPrompt]
    drive_status: DriveStatus | None


class NavLogModel(BaseModel):
    velocity: float
    yaw: float


class TimestampedLog(BaseModel):
    timestamp: dt


class OakdUnitreeNavLogModel(TimestampedLog):
    anytraverse: AnyTraverseLogModel
    navigation: NavLogModel


class OakdUnitreeNavLogs(BaseModel):
    logs: list[OakdUnitreeNavLogModel]


class LogManager:
    def __init__(self, out_dir: Path) -> None:
        self._out_dir = out_dir

        self._camera_video = self._create_video_writer(name="camera.avi")
        self._costmap_video = self._create_video_writer(
            name="costmap.avi", frame_size=(256, 256)
        )
        # self._trav_map_video = self._create_video_writer(name="trav_map.avi")
        # self._unc_map_video = self._create_video_writer(name="unc_map.avi")
        self.save_file = self._create_path_with_prefix(name="logs.json")
        self._logs = OakdUnitreeNavLogs(logs=[])

    def _create_path_with_prefix(self, name: str, is_dir: bool = False) -> Path:
        time_prefix = dt.now().strftime("%Y-%m-%d_%H%M%S__")
        path = self._out_dir / f"{time_prefix}{name}{'/' if is_dir else ''}"
        return path

    def _create_video_writer(
        self, name: str, fps: int = 30, frame_size: tuple[int, int] = (640, 480)
    ) -> cv2.VideoWriter:
        file = self._create_path_with_prefix(name=name)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")  # type: ignore
        writer = cv2.VideoWriter(
            file.as_posix(), fourcc=fourcc, fps=fps, frameSize=frame_size
        )
        return writer

    def log(
        self,
        state: AnyTraverseState,
        prompts: list[WeightedPrompt],
        velocity: float,
        yaw: float,
        costmap: Cv2Image,
        rgb: Cv2Image,
    ) -> None:
        log = OakdUnitreeNavLogModel(
            timestamp=dt.now(),
            anytraverse=AnyTraverseLogModel(
                roi_trav=state.trav_roi,
                roi_unc=state.unc_roi,
                drive_status=state.human_call,
                prompts=prompts,
            ),
            navigation=NavLogModel(velocity=velocity, yaw=yaw),
        )
        self._logs.logs.append(log)

        # Write to videos
        self._camera_video.write(rgb)
        self._costmap_video.write(costmap)
        # self._trav_map_video.write(
        #     (
        #         state.trav_map.unsqueeze(0)
        #         .expand(3, -1, -1)
        #         .permute(1, 2, 0)
        #         .cpu()
        #         .numpy()
        #         * 255
        #     ).astype(dtype=np.uint8)
        # )
        # self._unc_map_video.write(torch_tensor_to_cv_image(state.unc_map.cpu()))

    def save(self) -> Path:
        with open(self.save_file, "w") as file:
            file.write(self._logs.model_dump_json())
        return self.save_file
