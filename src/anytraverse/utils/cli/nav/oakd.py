import time
from pathlib import Path
from threading import Thread

import cv2
import depthai
import numpy as np
import torch
import typer
from numpy import typing as npt
from PIL import Image as PILImage
from pynput import keyboard
from roboticstoolbox import DstarPlanner
from typing_extensions import Annotated

from anytraverse.utils.cli.human_op.hoc_ctx import create_anytraverse_hoc_context
from anytraverse.utils.cli.human_op.io import (
    get_weighted_prompt_from_string as str_to_prompts,
)
from anytraverse.utils.cli.human_op.models import DriveStatus
from anytraverse.utils.helpers import DEVICE
from anytraverse.utils.helpers import mask_poolers as mask_poolers
from anytraverse.utils.helpers.grid_costmap import GridCostmap
from anytraverse.utils.helpers.log.oakd_nav_logger import LogManager
from anytraverse.utils.helpers.robots.unitree_zmq import UnitreeController
from anytraverse.utils.helpers.sensors.oakd import OakdCameraManager
from anytraverse.utils.pipelines.ws_human_op import AnyTraverseWebsocket

torch.set_default_device(device=DEVICE)
print(f"Using device: {DEVICE}")


CAM_TRANSFORM = np.array(
    [
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ]
)

EXIT: bool = False


def on_press(key: keyboard.KeyCode):
    global EXIT
    try:
        if key.char == "q":
            print("Key [Q] pressed.")
            EXIT = True
            return False
    except AttributeError:
        pass


def show_costmap_with_path(
    costmap: np.ndarray, path: list[tuple[int, int]], wait: int = 0
) -> npt.NDArray:
    img = (255 * costmap).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for x, y in path:
        if 0 <= x < costmap.shape[1] and 0 <= y < costmap.shape[0]:
            cv2.circle(img, (x, y), radius=1, color=(0, 0, 255), thickness=-1)

    img = cv2.resize(
        img,
        (img.shape[1] * 10, img.shape[0] * 10),
        interpolation=cv2.INTER_NEAREST,
    )
    # cv2.imshow("Costmap with Path", img)
    return img


def main(
    start_offset: Annotated[
        float,
        typer.Option(
            "--start-offset",
            "--so",
            help="How far ahead (in m) should the path planning start?",
        ),
    ],
    looakahead: Annotated[
        float,
        typer.Option(
            "--lookahead",
            "--la",
            help="How far into the path (in m) should the next waypoint be?",
        ),
    ],
    init_prompts: Annotated[
        str,
        typer.Option(
            "--prompts",
            "-p",
            help="Initial prompts for AnyTraverse, following prescribed syntax.",
        ),
    ],
    hoc_port: Annotated[
        int,
        typer.Option(help="Port to start the human operator call websocket server on."),
    ] = 7777,
    cost_temp: Annotated[float, typer.Option(help="Temperature for the costmap")] = 0.7,
    out_dir: Annotated[
        Path, typer.Option("-o", help="Output directory to store all logs")
    ] = Path("data/logs/nav/"),
):
    # Create AnyTraverse context
    anytraverse = create_anytraverse_hoc_context(
        init_prompts=str_to_prompts(prompts_str=init_prompts),
        mask_pooler=mask_poolers.ProbabilisticPooler,
    )

    # Start AnyTraverse WebSocket (for HOC UI, optional)
    ws_hoc = AnyTraverseWebsocket(anytraverse=anytraverse, port=hoc_port)
    Thread(target=ws_hoc.start).start()

    # DepthAI camera setup
    depthai_pipeline = depthai.Pipeline()
    oakd = OakdCameraManager(pipeline=depthai_pipeline)

    # Costmap
    costmap = GridCostmap(x_bound=8, y_bound=5, resolution=0.15)

    # Robot controller (ZMQ)
    robot = UnitreeController(hostname="localhost", port=6969)
    robot.connect()

    # Create keyboard listener
    kb_listener = keyboard.Listener(on_press=on_press)  # type: ignore
    kb_listener.start()

    # Create the log manager
    log_manager = LogManager(out_dir=out_dir)

    # Start OAK-D device
    with depthai.Device(pipeline=depthai_pipeline) as depthai_device:
        oakd.setup_with_device(device=depthai_device)

        while not EXIT:
            image, pointcloud = oakd.read_img_and_pointcloud()
            pil_image = PILImage.fromarray(image)
            anytraverse_state = anytraverse.run_next(frame=pil_image)

            print(anytraverse_state.trav_map.device)

            # Convert traversal map to costs
            costs = 1 - (
                anytraverse_state.trav_map.flatten().cpu().numpy().astype(np.float32)
            )

            # Apply camera transform
            pointcloud = (
                np.linalg.inv(CAM_TRANSFORM)
                @ np.vstack((pointcloud.T, np.ones(pointcloud.shape[0])))
            ).T[:, :3]

            # Update the costmap
            rows, cols = costmap.update_costmap(
                points=pointcloud, costs=costs, sigma=0.2, alpha=0.4
            )

            # Display original image
            cv2.imshow("Original Image", image)

            # Plan path using D*
            costmap_grid = costmap.get_grid()
            temperature = cost_temp
            plan_costmap = np.full_like(
                costmap_grid, fill_value=100 * np.exp(1 / temperature)
            )
            plan_costmap[rows, cols] = np.exp(costmap_grid[rows, cols] / temperature)

            d_star = DstarPlanner(
                costmap=plan_costmap,
                goal=(min(rows), costmap._width // 2)[::-1],
            )
            d_star.plan()

            start_point = (
                costmap._height - int(start_offset / costmap._resolution),
                costmap._width // 2,
            )[::-1]

            print("Start point:", start_point)

            path, _ = d_star.query(start=start_point)
            costmap_img = show_costmap_with_path(
                costmap=costmap_grid, path=path.tolist()
            )
            cv2.imshow("Costmap with path", costmap_img)

            # if len(path) > int(1.0 / costmap._resolution):
            # Select initial motion direction
            w0, w1 = path[[0, int(looakahead / costmap._resolution)]]
            w0[1], w1[1] = costmap._height - w0[1], costmap._height - w1[1]

            if anytraverse_state.human_call is not DriveStatus.OK:
                w1 = w0
                print(
                    f"HELP [{anytraverse_state.human_call}]! Human operator call required"
                )
                robot.stop_robot()
                time.sleep(1)

            print("Sending command to robot:", w0, w1)
            robot.send_command(start=w0, goal=w1)
            # if type(response) is dict and len(response.values()) == 2:
            #     velocity, yaw = response.values()
            # else:
            #     velocity, yaw = 0, 0

            cv2.waitKey(1)
            print(f"PROMPTS = {anytraverse.prompts}")

            log_manager.log(
                state=anytraverse_state,
                prompts=anytraverse.prompts,
                velocity=0,
                yaw=0,
                costmap=cv2.resize(
                    costmap_img, (256, 256), interpolation=cv2.INTER_NEAREST
                ).astype(dtype=np.uint8),
                rgb=cv2.resize(
                    image, (640, 480), interpolation=cv2.INTER_LANCZOS4
                ).astype(dtype=np.uint8),
            )

        cv2.destroyAllWindows()
        kb_listener.join()
        print("Saving data...")
        print(f"Saved logs to {log_manager.save()}")
        print("================ EXIT ANYTRAVERSE ================")
