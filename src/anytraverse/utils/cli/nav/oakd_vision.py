from abc import ABC, abstractmethod
from datetime import datetime as dt
from enum import Enum
from threading import Thread
from typing import Callable

import depthai
import numpy as np
import torch
from numpy import typing as npt
from oakd_sensor.utils.zeromq.pub import TimestampedCameraPublisher as FeedPublisher
from PIL import Image as PILImage
from rich.console import Console
from typing_extensions import override

from anytraverse.utils.cli.human_op.hoc_ctx import (
    AnyTraverseHOC_Context as AnyTraverse,
)
from anytraverse.utils.cli.human_op.hoc_ctx import (
    HumanOperatorControllerState as AnyTraverseState,
)
from anytraverse.utils.cli.human_op.hoc_ctx import (
    create_anytraverse_hoc_context,
)
from anytraverse.utils.cli.human_op.io import get_weighted_prompt_from_string
from anytraverse.utils.cli.human_op.models import DriveStatus
from anytraverse.utils.helpers import mask_poolers as mask_poolers
from anytraverse.utils.helpers.log.frame_logger import AnyTraverseLogger
from anytraverse.utils.helpers.robots.unitree_zmq import (
    RobotControlCommand,
    UnitreeZMQPublisher,
)
from anytraverse.utils.helpers.sensors.oakd import OakdCameraManager
from anytraverse.utils.pipelines.ws_human_op import AnyTraverseWebsocket

type FuncControlRobot[TControlCommand] = Callable[[TControlCommand], None]
type NpImage = npt.NDArray[np.uint8]


class NavigationState(Enum):
    FORWARD = "forward"
    TRAV_TURN_LEFT = "trav_turn_left"
    TRAV_TURN_RIGHT = "trav_turn_right"
    HOC_STOP = "hoc_stop"


class StateBasedNavigator[TState: Enum, TControlCommand](ABC):
    def __init__(
        self,
        control_robot: FuncControlRobot[TControlCommand],
        state_control_commands: dict[TState, TControlCommand],
        init_state: TState,
    ) -> None:
        self._control_robot: FuncControlRobot[TControlCommand] = control_robot
        self._state_control_commands: dict[TState, TControlCommand] = (
            state_control_commands
        )
        self._state: TState = init_state

    @property
    def state(self) -> TState:
        return self._state

    @state.setter
    def state(self, new_state: TState) -> TState:
        self._state = new_state
        return self._state

    @abstractmethod
    def update(self, *args, **kwargs) -> TState:
        pass

    def step(self) -> TControlCommand:
        control_command: TControlCommand = self._state_control_commands[self._state]
        self._control_robot(control_command)
        return control_command


class AnyTraverseStateBasedNavigator(
    StateBasedNavigator[NavigationState, RobotControlCommand]
):
    def __init__(
        self,
        anytraverse: AnyTraverse,
        control_robot: Callable[[RobotControlCommand], None],
        max_vel: float,
        yaw_speed: float,
        init_state: NavigationState = NavigationState.FORWARD,
    ) -> None:
        super().__init__(
            control_robot=control_robot,
            state_control_commands={
                NavigationState.FORWARD: {"velocity": [max_vel, 0.0], "yaw_speed": 0.0},
                NavigationState.HOC_STOP: {"velocity": [0.0, 0.0], "yaw_speed": 0.0},
                NavigationState.TRAV_TURN_LEFT: {
                    "velocity": [0.0, 0.0],
                    "yaw_speed": -yaw_speed,
                },
                NavigationState.TRAV_TURN_RIGHT: {
                    "velocity": [0.0, 0.0],
                    "yaw_speed": yaw_speed,
                },
            },
            init_state=init_state,
        )
        self._anytraverse = anytraverse
        self._max_vel = max_vel
        self._yaw_speed = yaw_speed

    def _set_velolcity(self, velocity: float) -> None:
        self._state_control_commands[NavigationState.FORWARD].update(
            {"velocity": [velocity, 0.0]}
        )

    @override
    def update(self, anytraverse_state: AnyTraverseState) -> NavigationState:
        roi_trav_map = self._anytraverse._roi_checker._get_roi(
            mask=anytraverse_state.trav_map
        )
        roi_unc_map = self._anytraverse._roi_checker._get_roi(
            mask=anytraverse_state.unc_map
        )
        roi_not_certain_untrav_map = 1 - (1 - roi_trav_map) * (1 - roi_unc_map)
        if roi_not_certain_untrav_map.mean() < 0.5 and not (
            self.state is NavigationState.TRAV_TURN_LEFT
            or self.state is NavigationState.TRAV_TURN_RIGHT
        ):
            turn_mask = torch.ones_like(roi_not_certain_untrav_map)
            turn_mask[:, : turn_mask.shape[1] // 2] = -1
            self.state = (
                NavigationState.TRAV_TURN_LEFT
                if (roi_not_certain_untrav_map * turn_mask).mean() < 0
                else NavigationState.TRAV_TURN_RIGHT
            )
        else:
            if (
                anytraverse_state.human_call is DriveStatus.UNK_ROI_OBJ
                or anytraverse_state.human_call is DriveStatus.UNSEEN_SCENE
            ):
                self.state = NavigationState.HOC_STOP
            else:
                self._set_velolcity(velocity=anytraverse_state.trav_roi * self._max_vel)
                self.state = NavigationState.FORWARD

        return self.state


class AnyTraverseVisionNavigator:
    """
    Vision-only navigator using the AnyTraverse pipeline.
    """

    def __init__(
        self,
        anytraverse: AnyTraverse,
        control_robot: Callable[[RobotControlCommand], None],
        max_vel: float,
        yaw_speed: float,
        console: Console,
        feed_publisher: FeedPublisher,
    ) -> None:
        self._console: Console = console
        self._feed_publisher: FeedPublisher = feed_publisher

        self._anytraverse: AnyTraverse = anytraverse
        self._control_robot: Callable[[RobotControlCommand], None] = control_robot

        # Create the AnyTraverse human operator websocket server
        with self._console.status("Creating HOC server..."):
            self._ws_hoc: AnyTraverseWebsocket = AnyTraverseWebsocket(
                anytraverse=anytraverse, port=7777
            )
        self._console.log("HOC server created", style="green")
        self._ws_thread: Thread = Thread(target=self._ws_hoc.start)

        # Create the OAK-D camera manager
        depthai_pipeline: depthai.Pipeline = depthai.Pipeline()
        self._oakd: OakdCameraManager = OakdCameraManager(pipeline=depthai_pipeline)

        # Create the logger
        self._logger: AnyTraverseLogger = AnyTraverseLogger(
            save_dir="data/nav_vision", fps=10
        )

        # Initialize the navigator
        self._navigator: AnyTraverseStateBasedNavigator = (
            AnyTraverseStateBasedNavigator(
                anytraverse=anytraverse,
                control_robot=control_robot,
                max_vel=max_vel,
                yaw_speed=yaw_speed,
            )
        )

    def start(self) -> None:
        # Start the websocket server for human operator calls
        with self._console.status(
            f"Starting HOC server on ws://{self._ws_hoc._hostname}:{self._ws_hoc._port}"
        ):
            self._ws_thread.start()

        # Start the OAK-D camera
        with depthai.Device(pipeline=self._oakd.pipeline) as depthai_device:
            self._oakd.setup_with_device(device=depthai_device)

            try:
                while True:
                    image, _ = self._oakd.read_img_and_pointcloud()
                    pil_image = PILImage.fromarray(image)
                    anytraverse_state = self._anytraverse.run_next(frame=pil_image)
                    nav_state = self._navigator.update(
                        anytraverse_state=anytraverse_state
                    )
                    self._console.log(
                        f"Navigation state: [bold green]{nav_state.value.upper()}[/]"
                    )
                    logged_frame = self._logger.add_frame(
                        image=pil_image,
                        trav_map=anytraverse_state.trav_map,
                        unc_map=anytraverse_state.unc_map,
                    )
                    self._feed_publisher.publish(logged_frame, dt.now())
                    control_command = self._navigator.step()
                    vel, yaw = (
                        control_command["velocity"][0],
                        np.rad2deg(control_command["yaw_speed"]),
                    )
                    self._console.log(
                        f"Sent control: vel={vel:.3f} m/s ; yaw={yaw:.3f} deg/s"
                    )
            except KeyboardInterrupt:
                with self._console.status("Closing AnyTraverse vision navigator..."):
                    self._ws_thread.join()
                    self._logger.close()
                self._console.log(
                    f"Logs stored at [magenta]{self._logger.log_path}[/] and [magenta]{self._logger.video_path}[/]"
                )
                self._console.log(
                    "Completed AnyTraverse vision navigation", style="green"
                )


def main():
    console = Console()
    with console.status("Creating AnyTraverse pipeline..."):
        anytraverse = create_anytraverse_hoc_context(
            init_prompts=get_weighted_prompt_from_string(
                console.input("Enter init promtps: ")
            ),
            mask_pooler=mask_poolers.ProbabilisticPooler,
        )
        anytraverse._thresholds["ref_sim"] = 0.7
    console.log("Created AnyTraverse pipeline")

    # Create the navigator
    robot = UnitreeZMQPublisher()

    def control_robot(control_command: RobotControlCommand) -> None:
        robot.send(topic="cmd_vel", message=control_command)

    navigator = AnyTraverseVisionNavigator(
        anytraverse=anytraverse,
        control_robot=control_robot,
        max_vel=1.0,
        yaw_speed=np.deg2rad(45),
        console=console,
        feed_publisher=FeedPublisher(port=8000, topic="/anytraverse"),
    )
    navigator.start()


if __name__ == "__main__":
    main()
