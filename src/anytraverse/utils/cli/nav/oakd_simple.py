from datetime import datetime as dt
from threading import Thread

import cv2
import depthai

# import imutils
import numpy as np
import torch
from oakd_sensor.utils.zeromq.pub import TimestampedCameraPublisher as FeedPublisher
from PIL import Image as PILImage
from rhino.main import Rhino

# from roboticstoolbox import DstarPlanner
from anytraverse.utils.cli.human_op.hoc_ctx import create_anytraverse_hoc_context
from anytraverse.utils.cli.human_op.io import get_weighted_prompt_from_string
from anytraverse.utils.cli.human_op.models import DriveStatus
from anytraverse.utils.helpers import DEVICE
from anytraverse.utils.helpers import mask_poolers as mask_poolers
from anytraverse.utils.helpers.log.frame_logger import AnyTraverseLogger

# from anytraverse.utils.helpers.robots.unitree_zmq import UnitreeZMQPublisher
from anytraverse.utils.helpers.sensors.oakd import OakdCameraManager

# from anytraverse.utils.helpers.grid_costmap import GridCostmap
from anytraverse.utils.pipelines.ws_human_op import AnyTraverseWebsocket


torch.set_default_device(device=DEVICE)
print(f"Using device: {DEVICE}")


CONNECTION_STR: str = "/dev/tty.usbmodem1101"
THROTTLE_CHANNEL = 3
STEERING_CHANNEL = 1

try:
    rhino_controller = Rhino(
        connection_string=CONNECTION_STR,
        log_dir="logs",
        file_name_prefix="drive_example",
        throttle_channel=THROTTLE_CHANNEL,
        steering_channel=STEERING_CHANNEL,
    )
except Exception as e:
    print(
        "Could not connect to Rhino controller. Make sure the connection string is correct."
    )
    print(e)
    exit(1)


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
    # init_prompts = get_weighted_prompt_from_string(input(">>> Enter initial prompts: "))
    init_prompts = get_weighted_prompt_from_string("tree: -1; rock: -1")
    anytraverse = create_anytraverse_hoc_context(
        init_prompts=init_prompts,
        mask_pooler=mask_poolers.ProbabilisticPooler,
    )
    anytraverse._thresholds["ref_sim"] = 0.75
    anytraverse._thresholds["unc_roi"] = 0.35

    anytraverse._roi_checker.y_bounds = (0.50, 1.00)
    anytraverse._roi_checker.x_bounds = (0.25, 0.75)

    # Start AnyTraverse WebSocket (for HOC UI, optional)
    ws_hoc = AnyTraverseWebsocket(anytraverse=anytraverse, port=7777)
    ws_thread = Thread(target=ws_hoc.start)
    ws_thread.start()

    # Feed publisher with ZMQ
    feed_pub = FeedPublisher(port=8000, topic="/anytraverse")

    # DepthAI camera setup
    depthai_pipeline = depthai.Pipeline()
    oakd = OakdCameraManager(pipeline=depthai_pipeline)

    # Costmap
    # costmap = GridCostmap(x_bound=8, y_bound=5, resolution=0.15)

    # Robot controller (ZMQ)
    # robot_command_publisher = UnitreeZMQPublisher()

    # Logger
    frame_logger = AnyTraverseLogger(save_dir="data/nav_simple", fps=10)

    # Start OAK-D device
    with depthai.Device(pipeline=depthai_pipeline) as depthai_device:
        oakd.setup_with_device(device=depthai_device)
        TURNING_TO_AVOID = False
        TURN_DIR = -1

        key = ""
        while key != "g":
            key = input("Type `g` and press [ENTER] to start ")

        try:
            while True:
                with ws_hoc.lock:
                    image, _ = oakd.read_img_and_pointcloud()
                    pil_image = PILImage.fromarray(image)
                    anytraverse_state = anytraverse.run_next(frame=pil_image)

                    print(
                        f"Frames in history: {len(anytraverse._scene_prompt_store._store)}",
                        end="\r",
                    )
                    print(f"Status: {anytraverse_state.human_call.value}", end="\r")

                    velocity: float = 0.0
                    yaw_speed: float = 0.0

                    roi_trav_thresh: float = 0.5
                    unc_roi_thresh: float = 0.25

                    # if (
                    #     anytraverse_state.human_call is DriveStatus.UNSEEN_SCENE
                    #     or anytraverse_state.human_call is DriveStatus.UNK_ROI_OBJ
                    # ):
                    if anytraverse_state.human_call is DriveStatus.UNK_ROI_OBJ:
                        TURNING_TO_AVOID = False
                        velocity = 0.0
                        yaw_speed = 0.0
                        print("----------- HUMAN OPERATOR CALLED --------------")
                    # elif anytraverse_state.trav_roi < 0.7:
                    elif anytraverse_state.trav_roi < 0.5:
                        if not TURNING_TO_AVOID:
                            trav_roi_map = anytraverse._roi_checker._get_roi(
                                anytraverse_state.trav_map
                            )
                            width = trav_roi_map.shape[1]
                            left_trav = trav_roi_map[:, width // 2].cpu().numpy().mean()
                            right_trav = (
                                trav_roi_map[:, width // 2 :].cpu().numpy().mean()
                            )
                            TURN_DIR = 1 if left_trav > right_trav else -1
                            # TURN_DIR = 1
                            TURNING_TO_AVOID = True
                            velocity = 0.0
                            # yaw_speed = TURN_DIR * np.deg2rad(60)
                            yaw_speed = TURN_DIR * 1.0
                    else:
                        TURNING_TO_AVOID = False
                        # velocity = 0.5 + (
                        #     0.3
                        #     * (1 - anytraverse_state.unc_roi)
                        #     * anytraverse_state.trav_roi
                        # )
                        velocity = 0.7
                        yaw_speed = 0.0

                    print(f"Trav roi = {anytraverse_state.trav_roi}", end="\r")

                    # velocity = 0.65
                    print(f"Sending command: vel={velocity}; yaw={yaw_speed}", end="\r")

                    try:
                        rhino_controller.send_velocity_cmd(
                            throttle=velocity, steering=yaw_speed
                        )
                    except Exception as e:
                        print("Could not send command to Rhino controller.")
                        print(e)
                        pass

                    # robot_command_publisher.send(
                    #     topic="cmd_vel",
                    #     message={"velocity": [velocity, 0.0], "yaw_speed": yaw_speed},
                    # )

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
                        text=f"Call: {anytraverse_state.human_call.value} | vel: {velocity}, steer: {yaw_speed} | {dict(anytraverse.prompts)}",
                    )
                    frame_logger.add_data(
                        trav_roi=anytraverse_state.trav_roi,
                        unc_roi=anytraverse_state.unc_roi,
                        prompts=anytraverse.prompts,
                    )
                    feed_pub.publish(frame, dt.now())

                # Display the frame
                cv2.imshow("AnyTraverse", frame)
                cv2.waitKey(1)
        except KeyboardInterrupt:
            print("Exiting AnyTraverse navigation...")
            frame_logger.close()
            ws_thread.join()


if __name__ == "__main__":
    main()
