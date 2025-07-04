"""
AnyTraverse Autonomous Navigation Pipeline

This module defines a clean, extensible structure for running autonomous
navigation with a traversability-aware model (AnyTraverse), sensor integration,
path planning, and robot actuation.

Components:
- OAK-D Pro camera input (via DepthAI)
- AnyTraverse model for traversability
- Grid costmap and D* planner
- Robot controller via ZMQ
- Optional WebSocket-based UI
"""

import numpy as np
import cv2
import depthai
import torch
from threading import Thread
from PIL import Image as PILImage
from roboticstoolbox import DstarPlanner
from numpy import typing as npt

from anytraverse.utils.helpers import DEVICE, mask_poolers
from anytraverse.utils.cli.human_op.io import get_weighted_prompt_from_string
from anytraverse.utils.cli.human_op.models import (
    HumanOperatorControllerState as AnyTraverseState,
    DriveStatus as AnyTraverseStatus,
)
from anytraverse.utils.cli.human_op.hoc_ctx import create_anytraverse_hoc_context
from anytraverse.utils.helpers.sensors.oakd import OakdCameraManager
from anytraverse.utils.helpers.grid_costmap import GridCostmap
from anytraverse.utils.pipelines.ws_human_op import AnyTraverseWebsocket
from anytraverse.utils.helpers.robots.unitree_zmq import (
    UnitreeZMQPublisher as UnitreeController,
    RobotCommand,
)
from anytraverse.utils.helpers.log.frame_logger import AnyTraverseLogger
from rich.console import Console
from oakd_sensor.utils.zeromq.pub import TimestampedCameraPublisher as FeedPublisher
from datetime import datetime as dt

CAM_TRANSFORM: npt.NDArray[np.float32] = np.array(
    [
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float32,
)


def visualize_path_on_costmap(
    costmap: npt.NDArray[np.float32], path: list[tuple[int, int]]
) -> npt.NDArray[np.uint8]:
    """
    Overlay a planned path on a grayscale costmap.

    Args:
        costmap: The 2D costmap.
        path: A list of (x, y) path waypoints.

    Returns:
        A BGR image with the path visualized.
    """
    image = 255 * costmap
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for x, y in path:
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(image, (x, y), radius=1, color=(0, 0, 255), thickness=-1)
    return image.astype(dtype=np.uint8)


class AnyTraverseNavigator:
    """Encapsulates the main autonomous navigation pipeline."""

    def __init__(self) -> None:
        self.device = DEVICE

        self._console = Console()

        torch.set_default_device(self.device)
        print(f"Using device: {self.device}")

        self._init_anytraverse()
        self._init_camera()
        self._init_costmap()
        self._init_robot()
        self._init_logger()
        self._init_websocket()

        self.TURNING_FOR_TRAV = False
        self.TURN_DIR = 0

        self.feed_publisher = FeedPublisher(port=8000, topic="/anytraverse")

    def _init_anytraverse(self) -> None:
        """Initializes the AnyTraverse context with user-defined prompts."""
        prompts = get_weighted_prompt_from_string(input(">>> Enter initial prompts: "))
        with self._console.status("Creating AnyTraverse pipeline..."):
            self.anytraverse = create_anytraverse_hoc_context(
                init_prompts=prompts,
                mask_pooler=mask_poolers.ProbabilisticPooler,
            )
        self.anytraverse._thresholds["ref_sim"] = 0.6
        self.anytraverse._thresholds["roi_unc"] = 0.7

        self._console.print("AnyTraverse pipeline created!")

    def _init_camera(self) -> None:
        """Initializes the DepthAI pipeline and OAK-D camera manager."""
        self.depthai_pipeline = depthai.Pipeline()
        self.oakd = OakdCameraManager(pipeline=self.depthai_pipeline)

    def _init_costmap(self) -> None:
        """Initializes the grid-based costmap."""
        self.costmap = GridCostmap(x_bound=8, y_bound=5, resolution=0.15)

    def _init_robot(self) -> None:
        """Initializes the ZMQ robot controller."""
        self.robot = UnitreeController()

    def _init_logger(self) -> None:
        """Initializes the frame and data logger."""
        self.logger = AnyTraverseLogger(save_dir="data/nav", fps=24)

    def _init_websocket(self) -> None:
        """Initializes the WebSocket server for HOC UI communication."""
        self.ws = AnyTraverseWebsocket(anytraverse=self.anytraverse, port=7777)
        self.ws_thread = Thread(target=self.ws.start)
        self.ws_thread.start()

    def run(self) -> None:
        """Starts the navigation loop and manages sensor input, planning, logging, and control."""
        with depthai.Device(pipeline=self.depthai_pipeline) as device:
            self.oakd.setup_with_device(device)

            try:
                while True:
                    self._tick()
            except KeyboardInterrupt:
                self._console.print("Exiting AnyTraverse navigation...")
                self.logger.close()
                self.ws_thread.join()

    def _tick(self) -> None:
        """Single iteration of sensing, perception, planning, and actuation."""
        image, pointcloud = self.oakd.read_img_and_pointcloud()
        pil_image = PILImage.fromarray(image)
        state: AnyTraverseState = self.anytraverse.run_next(frame=pil_image)

        costs: npt.NDArray[np.float32] = (
            1 - state.trav_map.flatten().cpu().numpy().astype(np.float32)
        )
        pointcloud = self._transform_pointcloud(pointcloud)

        rows, cols = self.costmap.update_costmap(
            points=pointcloud, costs=costs, sigma=0.35, alpha=0.4
        )

        plan_costmap = self._create_plan_costmap(rows, cols)
        path = self._plan_path(plan_costmap, rows)

        self._log_and_display(image, state, path)
        self._send_command(path, state, pointcloud)

    def _transform_pointcloud(
        self, pc: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """
        Apply camera-to-world transformation to point cloud.

        Args:
            pc: Raw pointcloud (N, 3).

        Returns:
            Transformed pointcloud (N, 3).
        """
        return (
            np.linalg.inv(CAM_TRANSFORM) @ np.vstack((pc.T, np.ones(pc.shape[0])))
        ).T[:, :3]

    def _create_plan_costmap(
        self, rows: npt.NDArray[np.uint16], cols: npt.NDArray[np.uint16]
    ) -> npt.NDArray[np.float32]:
        """
        Create a smoothed costmap for planning.

        Args:
            rows: Row indices of valid traversability.
            cols: Column indices.

        Returns:
            A planning costmap with exponential scaling.
        """
        grid = self.costmap.get_grid()
        temperature = 0.5
        costmap = np.full_like(grid, fill_value=100 * np.exp(1 / temperature))
        costmap[rows, cols] = np.exp(grid[rows, cols] / temperature)
        return costmap

    def _plan_path(
        self, costmap: npt.NDArray[np.float32], rows: npt.NDArray[np.uint16]
    ) -> npt.NDArray[np.uint16]:
        """
        Run D* planning algorithm.

        Args:
            costmap: Planning costmap.
            rows: Row indices to determine goal.

        Returns:
            A path of (x, y) grid indices.
        """
        goal = (int(min(rows) * 0.75), self.costmap._width // 2)[::-1]
        dstar = DstarPlanner(costmap=costmap, goal=goal)
        dstar.plan()

        start_y = self.costmap._height - int(0.2 / self.costmap._resolution)
        start = (start_y, self.costmap._width // 2)[::-1]
        self._console.print("Start point:", start)

        path, _ = dstar.query(start=start)
        return path

    def _log_and_display(
        self, image: npt.NDArray[np.uint8], state, path: npt.NDArray[np.uint16]
    ) -> None:
        """
        Visualizes path on image and logs the data.

        Args:
            image: RGB camera frame.
            state: Traversability output from AnyTraverse.
            path: Planned path in grid coordinates.
        """
        grid = self.costmap.get_grid()
        vis = visualize_path_on_costmap(costmap=grid, path=path.tolist())
        vis = cv2.resize(
            vis, (image.shape[0], image.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        stacked = np.concatenate((image, vis), axis=1)

        log_frame = self.logger.add_frame(
            image=PILImage.fromarray(stacked),
            trav_map=state.trav_map.cpu(),
            unc_map=state.unc_map.cpu(),
        )
        cv2.imshow("Image and Path", log_frame)

        # Publish the frame
        height, width, _ = log_frame.shape
        req_width = 1600
        req_height = int(req_width * height / width)
        self.feed_publisher.publish(
            cv2.resize(
                log_frame,
                dsize=(req_width, req_height),
                interpolation=cv2.INTER_LANCZOS4,
            ).astype(dtype=np.uint8),
            dt.now(),
        )

        self.logger.add_data(
            trav_roi=state.trav_roi,
            unc_roi=state.unc_roi,
            prompts=self.anytraverse.prompts,
        )
        cv2.waitKey(1)

    def _send_command(
        self,
        path: npt.NDArray[np.uint16],
        state: AnyTraverseState,
        pc: npt.NDArray[np.float32],
    ) -> None:
        """
        Computes velocity and heading command and sends to robot.

        Args:
            path: Planned path.
            state: AnyTraverse output state.
            pc: Transformed pointcloud.
        """
        if len(path) <= int(1.0 / self.costmap._resolution):
            return

        # LOOKAHEAD = 0.3
        # w0, w1 = path[[0, int(LOOKAHEAD / self.costmap._resolution)]].tolist()
        # w0[1], w1[1] = self.costmap._height - w0[1], self.costmap._height - w1[1]
        # x0, y0 = w0
        # x1, y1 = w1
        # theta = np.arctan2(x1 - x0, y1 - y0)

        # Set velocity according to traversability in immediate vicinity
        MAX_SPEED = 1.0
        vel = MAX_SPEED * state.trav_roi

        # Set the yaw using the maximum deviation from straight path in the next LOOKAHEAD meters
        lookahead: float = 1.0
        start_point = path[0]  # The starting point (always the same though)
        # The whole path, `lookahead` meters ahead
        lookahead_path: npt.NDArray[np.uint16] = path[
            : int(lookahead / self.costmap._resolution)
        ]
        # The point with the maximum deviation from horizontal coord of start point
        max_y_dev_point = path[np.argmax(np.abs(lookahead_path[:, 1] - start_point[1]))]
        # Extract coordinates of start and maximum deviation point
        x0, y0 = start_point
        x1, y1 = max_y_dev_point
        # Calculate the angle to go from start to max dev point
        theta = np.arctan2(x1 - x0, y1 - y0)

        # Stop the robot for human operator calls
        if (
            state.human_call is AnyTraverseStatus.UNK_ROI_OBJ
            or state.human_call is AnyTraverseStatus.UNSEEN_SCENE
        ):
            vel, theta = 0.0, 0.0
            self._console.print("Stopping for HOC", style="red")
        # If no traversability, start turning
        elif state.human_call is AnyTraverseStatus.BAD_ROI:
            vel = 0.0
            self._console.print("[WARN] Low traversability in ROI", style="yellow")
            if not self.TURNING_FOR_TRAV:
                self.TURNING_FOR_TRAV = True
                self.TURN_DIR = -1 if theta < 0 else 1
            else:
                self._console.print("Turning for traversability...", style="dim yellow")

        # Else, good to go
        else:
            self.TURNING_FOR_TRAV = False
            self.TURN_DIR = 0
            self._console.print("OK", vel, theta)
            # theta = 0

        if self.TURNING_FOR_TRAV:
            theta = self.TURN_DIR * np.pi / 4

        command: RobotCommand = {
            "velocity": [vel, 0.0],
            "yaw_speed": np.clip(theta, -np.pi / 2, np.pi / 2),
        }
        self._console.print(
            f"Drive status: [bold cyan]{state.human_call.value.upper()}[/]"
        )
        self._console.print(
            f"Control command: velocity=[bold yellow]{vel} m/s[/]; yaw_speed=[bold blue]{np.rad2deg(theta)} deg/s[/]"
        )
        self.robot.send(topic="cmd_vel", message=command)


def main():
    navigator = AnyTraverseNavigator()
    navigator.run()


if __name__ == "__main__":
    navigator = AnyTraverseNavigator()
    navigator.run()
