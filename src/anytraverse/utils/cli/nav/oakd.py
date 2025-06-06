import numpy as np
from PIL import Image as PILImage
import depthai
import cv2

from roboticstoolbox import DstarPlanner

from anytraverse.utils.cli.human_op.hoc_ctx import create_anytraverse_hoc_context
from anytraverse.utils.helpers import mask_poolers as mask_poolers
from anytraverse.utils.helpers.sensors.oakd import OakdCameraManager
from anytraverse.utils.helpers.grid_costmap import GridCostmap
from anytraverse.utils.pipelines.ws_human_op import AnyTraverseWebsocket
from threading import Thread

# TODO Uncomment this and start working on robot control
# from anytraverse.utils.helpers.robots.unitree_go1 import CommandBuilder, RobotController


CAM_TRANSFORM = np.array(
    [
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ]
)


def show_costmap_with_path(
    costmap: np.ndarray, path: list[tuple[int, int]], wait: int = 0
):
    """
    Displays a 2D grid costmap with the path overlaid using OpenCV.

    Args:
        costmap: 2D NumPy array of float values (0.0 to 1.0).
        path: List of (x, y) grid indices.
        wait: Milliseconds to wait for keypress in `cv2.waitKey`. 0 = wait forever.
    """
    # Convert costmap to 8-bit grayscale image (invert: 1 -> white, 0 -> black)
    img = (255 * costmap).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Draw path in red (BGR: 0, 0, 255)
    for x, y in path:
        if 0 <= x < costmap.shape[1] and 0 <= y < costmap.shape[0]:
            cv2.circle(img, (x, y), radius=1, color=(0, 0, 255), thickness=-1)

    # Resize for better visibility
    scale = 10  # each cell becomes 10x10 pixels
    img = cv2.resize(
        img,
        (img.shape[1] * scale, img.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    cv2.imshow("Costmap with Path", img)


def main():
    # Create the AnyTraverse pipeline
    anytraverse = create_anytraverse_hoc_context(
        init_prompts=[("floor", 1.0), ("mat", -0.5), ("shoes", -0.8)],
        mask_pooler=mask_poolers.ProbabilisticPooler,
    )

    # AnyTraverse websocket handler server
    ws_hoc = AnyTraverseWebsocket(anytraverse=anytraverse, port=7777)
    thread = Thread(target=ws_hoc.start)
    thread.start()

    # Create the depthai pipeline
    depthai_pipeline = depthai.Pipeline()

    # Create the OAK-D camera manager
    oakd = OakdCameraManager(pipeline=depthai_pipeline)

    # Grid costmap
    costmap = GridCostmap(x_bound=8, y_bound=5, resolution=0.2)

    # Create the robot utils
    # command = CommandBuilder()
    # robot = RobotController()

    # Connect to the device
    with depthai.Device(pipeline=depthai_pipeline) as depthai_device:
        oakd.setup_with_device(device=depthai_device)

        while True:
            # Get the image and the pointcloud
            image, pointcloud = oakd.read_img_and_pointcloud()

            # Show the pointcloud shape
            # print(f"Pointcloud shape: {pointcloud.shape}")

            # Create PIL image for input into AnyTraverse
            pil_image = PILImage.fromarray(image)
            # ws_hoc.lock.acquire()
            anytraverse_state = anytraverse.run_next(frame=pil_image)
            prompts = anytraverse.prompts
            # ws_hoc.lock.release()

            # Print prompts
            # print(f"Prompts >>> {prompts}")

            costs = 1 - (
                anytraverse_state.trav_map.flatten()
                .cpu()
                .numpy()
                .astype(dtype=np.float32)
            )

            # Transform coords
            pointcloud = (
                np.linalg.inv(CAM_TRANSFORM)
                @ (
                    np.concatenate(
                        (pointcloud.T, np.ones((1, pointcloud.shape[0]))), axis=0
                    )
                )
            ).T[:, :3]

            # Update costmap
            rows, cols = costmap.update_costmap(
                points=pointcloud, costs=costs, sigma=0.1, alpha=0.3
            )

            # Show the original image
            cv2.imshow("Original Image", image)

            # Create costmap and plan path
            costmap_grid = costmap.get_grid()
            temperature: float = 0.7
            plan_costmap = np.full_like(
                costmap_grid, fill_value=100 * np.exp(1 / temperature)
            )
            plan_costmap[rows, cols] = np.exp(costmap_grid[rows, cols] / temperature)
            d_star = DstarPlanner(
                costmap=plan_costmap,
                goal=(int(1.00 / costmap._resolution), costmap._width // 2)[::-1],
            )
            d_star.plan()
            start_point = (
                costmap._height - int(0.5 / costmap._resolution),
                costmap._width // 2,
            )[::-1]
            path, _ = d_star.query(start=start_point)
            show_costmap_with_path(costmap=costmap_grid, path=path.tolist())

            # Navigate the robot
            w0, w1 = path[[0, int(1.0 / costmap._resolution)], ::-1] - np.array(
                [start_point]
            )
            w1 = w1 - w0

            # vel, yaw = CommandBuilder.compute_velocity(
            #     start=np.zeros_like(w1), target=w1
            # )
            # robot.send_command(velocity=vel, yaw_speed=yaw)

            # angle = np.rad2deg((np.arctan2(*(w1 - w0)[::-1]) - np.pi) % (2 * np.pi))
            # print(f"Turn {angle} deg")
            # print(f"Waypoints = {waypoints.tolist()}")
            # print(f"Turn by: {np.rad2deg(np.arctan2(waypoints))}")

            cv2.waitKey(1)


if __name__ == "__main__":
    main()
