import numpy as np
from numpy import typing as npt
from PIL import Image as PILImage

from anytraverse.utils.cli.human_op.hoc_ctx import (
    AnyTraverseHOC_Context as AnyTraverse,
    HumanOperatorControllerState as AnyTraverseState,
)
from anytraverse.utils.pipelines.ws_human_op import AnyTraverseWebsocket
from anytraverse.utils.helpers.sensors.oakd import OakdCameraManager
from anytraverse.utils.helpers.grid_costmap import GridCostmap
import depthai
from typing import TypedDict


class AnyTraverseCostmapOutput(TypedDict):
    anytraverse: AnyTraverseState


class AnyTraverseOakdCostmap:
    _anytraverse: AnyTraverse
    _ws_hoc: AnyTraverseWebsocket
    _oakd: OakdCameraManager
    _costmap_manager: GridCostmap
    _state: AnyTraverseState

    def __init__(
        self,
        anytraverse: AnyTraverse,
        grid_costmap: GridCostmap,
        hoc_hostname: str = "0.0.0.0",
        hoc_port: int = 6969,
    ) -> None:
        self._anytraverse = anytraverse

        # Initialize OAK-D camera
        depthai_pipeline = depthai.Pipeline()
        self._oakd = OakdCameraManager(pipeline=depthai_pipeline)
        with depthai.Device(pipeline=depthai_pipeline) as device:
            self._oakd.setup_with_device(device=device)

        # Setup human operator call websocket server
        self._ws_hoc = AnyTraverseWebsocket(
            anytraverse=self._anytraverse, hostname=hoc_hostname, port=hoc_port
        )

        # Setup costmap
        self._costmap_manager = grid_costmap

    @property
    def state(self) -> AnyTraverseState:
        return self._state

    @property
    def costmap(self) -> GridCostmap:
        return self._costmap_manager

    def start_hoc_server(self) -> None:
        self._ws_hoc.start()

    def step(self, cam_to_world_transform: npt.NDArray[np.float32]):
        rgb_frame, pointcloud = self._oakd.read_img_and_pointcloud(
            cam_to_world_transform=cam_to_world_transform
        )

        # Send image through AnyTraverse
        pil_image = PILImage.fromarray(rgb_frame)
        self._state = self._anytraverse.run_next(frame=pil_image)

        # Update the costmap
        self._costmap_manager.update_costmap(
            points=pointcloud,
            costs=self._state.trav_map.flatten().cpu().numpy().astype(dtype=np.float32),
        )
