import numpy as np
import torch
from anytraverse.utils.helpers.sensors.oakd import OakdCameraManager
from depthai import Device, Pipeline
from anytraverse import (
    create_anytraverse_hoc_context as create_anytraverse,
    mask_poolers as mask_poolers,
)
from anytraverse.utils.helpers import DEVICE
from PIL import Image

import queue
import threading
import time

from multiprocessing import shared_memory as shm


torch.set_default_device(device=DEVICE)

# Thread-safe queue for human commands
human_command_queue = queue.Queue()


def human_operator_input_loop():
    """
    Runs in a separate thread to get commands from the console.
    Type "exit" to quit.
    """
    while True:
        try:
            cmd = input("> ").strip()
            if cmd.lower() == "exit":
                break
            if cmd:
                human_command_queue.put(cmd)
        except EOFError:
            break  # Handles Ctrl+D


def main():
    input_thread = threading.Thread(target=human_operator_input_loop, daemon=True)
    input_thread.start()

    pipeline = Pipeline()

    oakd_manager = OakdCameraManager(
        pipeline=pipeline,
        rgb_stream_name="rgb",
        pointcloud_stream_name="points",
        min_depth=0.25,
        max_depth=10.0,
    )

    shm_name = "oakd_pointcloud"
    shm_size = np.dtype(np.float32).itemsize * (230400 * 4) + 1
    shm_pointcloud = shm.SharedMemory(name=shm_name, create=True, size=shm_size)
    buffer = shm_pointcloud.buf

    with Device(pipeline) as device:
        oakd_manager.setup_with_device(device=device)

        anytraverse = create_anytraverse(
            init_prompts=[("floor", 0.95), ("chair", -0.85)],
            mask_pooler=mask_poolers.ProbabilisticPooler,
        )

        try:
            while True:
                rgb, points = oakd_manager.read_img_and_pointcloud()

                # Process all human inputs
                while not human_command_queue.empty():
                    cmd = human_command_queue.get()
                    anytraverse.human_call_with_syntax(prompts_str=cmd)

                state = anytraverse.run_next(frame=Image.fromarray(rgb))

                pointcloud_buf_np = np.ndarray(
                    shape=(640 * 360, 4), dtype=np.float32, buffer=buffer[1:]
                )
                pointcloud_data = np.hstack(
                    (
                        points,
                        state.trav_map.cpu().numpy().astype(np.float32).reshape(-1, 1),
                    )
                )
                pointcloud_buf_np[:] = pointcloud_data[:]
                buffer[0] = 1  # mark buffer ready

                # Optional: Do your own printing here
                # e.g., print(f"Traversability ROI: {state.trav_roi}")

                time.sleep(0.01)  # Reduce CPU usage

        finally:
            shm_pointcloud.close()
            shm_pointcloud.unlink()


if __name__ == "__main__":
    main()
