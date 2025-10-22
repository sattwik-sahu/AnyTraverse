from datetime import datetime as dt
from enum import Enum
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
from numpy import typing as npt
import torch
import typer
from oakd_sensor.utils.zeromq.pub import TimestampedCameraPublisher as FeedPublisher
from oakd_vio_zmq.subscribe import Subscriber as OAKDSubscriber
from PIL import Image as PILImage
from rich.console import Console
from typing_extensions import Annotated

from anytraverse.utils.cli.human_op.hoc_ctx import create_anytraverse_hoc_context
from anytraverse.utils.cli.human_op.io import get_weighted_prompt_from_string
from anytraverse.utils.cli.human_op.models import DriveStatus
from anytraverse.utils.helpers import DEVICE
from anytraverse.utils.helpers import mask_poolers as mask_poolers
from anytraverse.utils.helpers.log.frame_logger import AnyTraverseLogger
from anytraverse.utils.pipelines.ws_human_op import AnyTraverseWebsocket
from scipy.spatial.transform import Rotation as R


def tf_to_cartesian(
    tf: npt.NDArray[np.float64],
) -> tuple[float, float, float, float, float, float]:
    x, y, z = tf[:3, 3].tolist()
    rot = tf[:3, :3]
    roll, pitch, yaw = R.from_matrix(rot).as_euler("xyz", degrees=False).tolist()
    return x, y, z, roll, pitch, yaw


class Pooler(Enum):
    WEIGHTED_MAX = "wmax"
    PROBABILISTIC = "proba"


def run(
    prompts: Annotated[
        str,
        typer.Option(
            prompt="Enter initial prompts", help="Initial prompts for AnyTraverse"
        ),
    ],
    oakd_stream_name: Annotated[
        str,
        typer.Option(
            "--oakd",
            help="Name of the OAKD sensor data publisher stream",
            prompt="Please enter OAK-D sensor stream name",
        ),
    ],
    ws_port: Annotated[
        int, typer.Option(help="Port to start HOC websocket server on")
    ] = 7777,
    feed_port: Annotated[
        int, typer.Option(help="The port to publish the annotated AnyTraverse feed on")
    ] = 8080,
    pooler: Annotated[
        Pooler, typer.Option(help="The map pooler to use with AnyTraverse")
    ] = Pooler.PROBABILISTIC,
    thresh_sim: Annotated[
        float, typer.Option(help="The scene similarity threshold")
    ] = 0.5,
    thresh_unc: Annotated[
        float, typer.Option(help="The ROI uncertainty threshold")
    ] = 0.5,
    roi_x_bounds: Annotated[
        tuple[float, float],
        typer.Option(
            "--roi-xb", help="ROI horizontal bounds relative to image size, of ROI"
        ),
    ] = (0.67, 1.00),
    roi_y_bounds: Annotated[
        tuple[float, float],
        typer.Option("--roi-yb", help="ROI vertical bounds relative to image size"),
    ] = (0.33, 0.67),
    save_dir: Annotated[Path, typer.Option(help="Directory to save logs")] = Path(
        "data/nav"
    ),
) -> None:
    console = Console()

    # Choose mask pooler
    match pooler:
        case Pooler.WEIGHTED_MAX:
            mask_pooler = mask_poolers.WeightedMaxPooler
        case Pooler.PROBABILISTIC:
            mask_pooler = mask_poolers.ProbabilisticPooler

    # Create AnyTraverse pipeline
    with console.status("Initializing AnyTraverse pipeline..."):
        anytraverse = create_anytraverse_hoc_context(
            init_prompts=get_weighted_prompt_from_string(prompts_str=prompts),
            mask_pooler=mask_pooler,
        )
    console.log("Loaded AnyTraverse")

    # Set custom pipeline parameters
    with console.status("Setting custom parameters for AnyTraverse..."):
        anytraverse._thresholds["ref_sim"] = thresh_sim
        anytraverse._thresholds["roi_unc"] = thresh_unc
        anytraverse._roi_checker.x_bounds = roi_x_bounds
        anytraverse._roi_checker.y_bounds = roi_y_bounds
    console.log("AnyTraverse custom parameters set!")

    # Start AnyTraverse websocket server for human operator calls
    try:
        ws_hoc = AnyTraverseWebsocket(anytraverse=anytraverse, port=ws_port)
        ws_thread = Thread(target=ws_hoc.start)
        ws_thread.start()
        console.log(
            f"AnyTraverse human operator websocket server started on tcp://0.0.0.0:{ws_port}"
        )
    except Exception as ex:
        console.log(
            "[red]Error starting AnyTraverse Human Operator websocket server![/]"
        )
        raise ex

    # Setup logger and feed publisher
    logger = AnyTraverseLogger(save_dir=save_dir, fps=12)
    feed_pub = FeedPublisher(port=feed_port, topic="/anytraverse")

    # Setup OAK-D sensor
    oakd = OAKDSubscriber(stream_name=oakd_stream_name)
    with console.status("Connecting to OAK-D data stream..."):
        try:
            oakd.connect()
        except Exception as ex:
            console.log("[red]Could not connect to OAK-D[/]")
            raise ex
    console.log("Connected to OAK-D data stream")

    # Main navigation loop
    try:
        while True:
            with ws_hoc.lock:
                # Get data from sensor data stream
                oakd_data = oakd.get_next()

                # Wait if no data
                if oakd_data is None:
                    with console.status("Waiting for OAK-D data stream..."):
                        while oakd_data is None:
                            oakd_data = oakd.get_next()

                # Extract data
                rgb, pointcloud, R_cw = (
                    oakd_data.rgb,
                    oakd_data.pointcloud,
                    oakd_data.transform,
                )

                console.log(f"Image: {rgb.shape}")
                console.log(f"Pointcloud: {pointcloud.shape}")
                x, y, z, roll, pitch, yaw = tf_to_cartesian(tf=R_cw)
                console.log(dict(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw))

                # Pass image through AnyTraverse pipeline
                rgb_pil = PILImage.fromarray(rgb)
                anytraverse_state = anytraverse.run_next(frame=rgb_pil)

                # Get traversabilty and uncertainty map on image
                trav_map = anytraverse_state.trav_map.cpu()
                unc_map = anytraverse_state.unc_map.cpu()

                # Log the frame
                if logger.writer is None:
                    logger._init_writer(
                        height=rgb.shape[0],
                        width=rgb.shape[1] * 3,
                    )
                frame = logger.add_frame(
                    image=rgb_pil,
                    trav_map=trav_map,
                    unc_map=unc_map,
                    text=(
                        f"x: {x:.2f}, y: {y:.2f}, z: {z:.2f}, "
                        f"roll: {np.degrees(roll):.2f} deg, pitch: {np.degrees(pitch):.2f} deg, yaw: {np.degrees(yaw):.2f} deg"
                    ),
                )
                feed_pub.publish(frame, dt.now())

            cv2.imshow("AnyTraverse", frame)
            cv2.waitKey(1)

            # Create costmap

    except KeyboardInterrupt:
        console.log("Exiting AnyTraverse navigation...")
        logger.close()
        ws_thread.join()
        oakd.close()
        console.log("[cyan]Done.[/]")
