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
from rich.live import Live
from rich.table import Table
from rich.console import Console
from rich import box

console = Console()


torch.set_default_device(device=DEVICE)
print(f"Using device: {DEVICE}")


CONNECTION_STR: str = "/dev/tty.usbmodem1101"
THROTTLE_CHANNEL = 3
STEERING_CHANNEL = 1

# try:
#     rhino_controller = Rhino(
#         connection_string=CONNECTION_STR,
#         log_dir="logs",
#         file_name_prefix="drive_example",
#         throttle_channel=THROTTLE_CHANNEL,
#         steering_channel=STEERING_CHANNEL,
#     )
# except Exception as e:
#     print(
#         "Could not connect to Rhino controller. Make sure the connection string is correct."
#     )
#     print(e)
#     exit(1)


CAM_TRANSFORM = np.array(
    [
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ]
)


def main():
    print("Creating AnyTraverse Pipeline...")
    init_prompts = get_weighted_prompt_from_string("grass: 1.0; tree: -1; rock: -1")
    anytraverse = create_anytraverse_hoc_context(
        init_prompts=init_prompts,
        mask_pooler=mask_poolers.ProbabilisticPooler,
    )
    anytraverse._thresholds["ref_sim"] = 0.75
    anytraverse._roi_checker.y_bounds = (0.50, 1.00)
    anytraverse._roi_checker.x_bounds = (0.25, 0.75)

    ws_hoc = AnyTraverseWebsocket(anytraverse=anytraverse, port=7777)
    ws_thread = Thread(target=ws_hoc.start)
    ws_thread.start()

    feed_pub = FeedPublisher(port=8000, topic="/anytraverse")

    depthai_pipeline = depthai.Pipeline()
    oakd = OakdCameraManager(pipeline=depthai_pipeline)

    frame_logger = AnyTraverseLogger(save_dir="data/nav_simple", fps=10)

    # ---- RICH TABLE SETUP ----
    def make_status_table() -> Table:
        table = Table(title="AnyTraverse Navigation Monitor", box=box.ROUNDED)
        table.add_column("Metric", justify="right", style="bold cyan")
        table.add_column("Value", justify="left", style="bold white")
        return table

    with depthai.Device(pipeline=depthai_pipeline) as depthai_device:
        oakd.setup_with_device(device=depthai_device)
        TURNING_TO_AVOID = False
        TURN_DIR = -1

        key = ""
        while key != "g":
            key = input("Type `g` and press [ENTER] to start ")

        frame_count = 0

        with Live(make_status_table(), refresh_per_second=4, console=console) as live:
            try:
                while True:
                    with ws_hoc.lock:
                        image, _ = oakd.read_img_and_pointcloud()
                        pil_image = PILImage.fromarray(image)
                        anytraverse_state = anytraverse.run_next(frame=pil_image)

                        frame_count += 1
                        velocity: float = 0.0
                        yaw_speed: float = 0.0
                        roi_trav_thresh: float = 0.5
                        unc_roi_thresh: float = 0.25

                        if (
                            anytraverse_state.human_call is DriveStatus.UNSEEN_SCENE
                            or anytraverse_state.human_call is DriveStatus.UNK_ROI_OBJ
                        ):
                            TURNING_TO_AVOID = False
                            velocity = 0.0
                            yaw_speed = 0.0
                        elif anytraverse_state.trav_roi < roi_trav_thresh:
                            if not TURNING_TO_AVOID:
                                trav_roi_map = anytraverse._roi_checker._get_roi(
                                    anytraverse_state.trav_map
                                )
                                width = trav_roi_map.shape[1]
                                left_trav = (
                                    trav_roi_map[:, : width // 2].cpu().numpy().mean()
                                )
                                right_trav = (
                                    trav_roi_map[:, width // 2 :].cpu().numpy().mean()
                                )
                                TURN_DIR = 1 if left_trav > right_trav else -1
                                TURNING_TO_AVOID = True
                                velocity = 0.0
                                yaw_speed = TURN_DIR * 1.0
                        else:
                            TURNING_TO_AVOID = False
                            velocity = 0.5 + (
                                0.3
                                * (1 - anytraverse_state.unc_roi)
                                * anytraverse_state.trav_roi
                            )
                            yaw_speed = 0.0

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

                        # ---- UPDATE RICH TABLE ----
                        table = make_status_table()
                        table.add_row("Frames", str(frame_count))
                        table.add_row(
                            "Traversability ROI", f"{anytraverse_state.trav_roi:.3f}"
                        )
                        table.add_row(
                            "Uncertainty ROI", f"{anytraverse_state.unc_roi:.3f}"
                        )
                        table.add_row(
                            "Drive Status", str(anytraverse_state.human_call.value)
                        )
                        table.add_row("Velocity", f"{velocity:.3f}")
                        table.add_row("Yaw Speed", f"{yaw_speed:.3f}")
                        table.add_row("Turning", str(TURNING_TO_AVOID))
                        table.add_row("Turn Dir", str(TURN_DIR))
                        table.add_row("Prompts", str(anytraverse.prompts))
                        live.update(table)

                        # Display the frame
                        cv2.imshow("AnyTraverse", frame)
                        cv2.waitKey(1)
            except KeyboardInterrupt:
                print("Exiting AnyTraverse navigation...")
                frame_logger.close()
                ws_thread.join()


if __name__ == "__main__":
    main()
