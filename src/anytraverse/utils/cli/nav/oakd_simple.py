import numpy as np
import imutils
from PIL import Image as PILImage
import depthai
import cv2
import torch

# from roboticstoolbox import DstarPlanner

from anytraverse.utils.cli.human_op.hoc_ctx import create_anytraverse_hoc_context
from anytraverse.utils.cli.human_op.io import get_weighted_prompt_from_string
from anytraverse.utils.cli.human_op.models import DriveStatus
from anytraverse.utils.helpers import mask_poolers as mask_poolers
from anytraverse.utils.helpers.sensors.oakd import OakdCameraManager

# from anytraverse.utils.helpers.grid_costmap import GridCostmap
from anytraverse.utils.pipelines.ws_human_op import AnyTraverseWebsocket
from threading import Thread
from anytraverse.utils.helpers.robots.unitree_zmq import (
    UnitreeZMQPublisher,
    RobotCommand,
)
from anytraverse.utils.helpers import DEVICE
from anytraverse.utils.helpers.log.frame_logger import AnyTraverseLogger


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


def main():
    # Create AnyTraverse context
    print("Creating AnyTraverse Pipeline...")
    init_prompts = get_weighted_prompt_from_string(input(">>> Enter initial prompts: "))
    anytraverse = create_anytraverse_hoc_context(
        init_prompts=init_prompts,
        mask_pooler=mask_poolers.ProbabilisticPooler,
    )
    anytraverse._thresholds["ref_sim"] = 0.75

    # Start AnyTraverse WebSocket (for HOC UI, optional)
    ws_hoc = AnyTraverseWebsocket(anytraverse=anytraverse, port=7777)
    ws_thread = Thread(target=ws_hoc.start)
    ws_thread.start()

    # DepthAI camera setup
    depthai_pipeline = depthai.Pipeline()
    oakd = OakdCameraManager(pipeline=depthai_pipeline)

    # Costmap
    # costmap = GridCostmap(x_bound=8, y_bound=5, resolution=0.15)

    # Robot controller (ZMQ)
    robot_command_publisher = UnitreeZMQPublisher()

    # Logger
    frame_logger = AnyTraverseLogger(save_dir="data/nav", fps=10)

    # Start OAK-D device
    with depthai.Device(pipeline=depthai_pipeline) as depthai_device:
        oakd.setup_with_device(device=depthai_device)
        TURNING_TO_AVOID = False
        TURN_DIR = -1

        try:
            while True:
                with ws_hoc.lock:
                    image, _ = oakd.read_img_and_pointcloud()
                    pil_image = PILImage.fromarray(image)
                    anytraverse_state = anytraverse.run_next(frame=pil_image)

                    print(
                        f"Frames in history: {len(anytraverse._scene_prompt_store._store)}"
                    )
                    print(f"Status: {anytraverse_state.human_call.value}")

                    velocity: float = 0.0
                    yaw_speed: float = 0.0

                    if (
                        anytraverse_state.unc_roi > 0.25
                        or anytraverse_state.trav_roi < 0.5
                    ):
                        if not TURNING_TO_AVOID:
                            TURN_DIR = np.random.choice([-1, 1])
                            TURNING_TO_AVOID = True
                            velocity = 0.0
                            yaw_speed = TURN_DIR * np.deg2rad(45)
                    elif anytraverse_state.human_call is DriveStatus.UNSEEN_SCENE:
                        TURNING_TO_AVOID = False
                        velocity = 0.0
                        yaw_speed = 0.0
                    else:
                        TURNING_TO_AVOID = False
                        velocity = 0.5
                        yaw_speed = 0.0

                    print(f"Sending command: vel={velocity}; yaw={yaw_speed}")
                    robot_command_publisher.send(
                        topic="cmd_vel",
                        message={"velocity": [velocity, 0.0], "yaw_speed": yaw_speed},
                    )

                    # Log the traversability and uncertainty maps
                    trav_map = anytraverse_state.trav_map.cpu()
                    unc_map = anytraverse_state.unc_map.cpu()
                    if frame_logger.writer is None:
                        frame_logger._init_writer(
                            height=image.shape[0],
                            width=image.shape[1] * 3,
                        )
                    frame = frame_logger.add_frame(
                        image=pil_image,
                        trav_map=trav_map,
                        unc_map=unc_map,
                    )
                    frame_logger.add_data(
                        trav_roi=anytraverse_state.trav_roi,
                        unc_roi=anytraverse_state.unc_roi,
                        prompts=anytraverse.prompts,
                    )

                # Display the frame
                cv2.imshow("AnyTraverse", frame)
                cv2.waitKey(1)
        except KeyboardInterrupt:
            print("Exiting AnyTraverse navigation...")
            frame_logger.close()
            ws_thread.join()


if __name__ == "__main__":
    main()
