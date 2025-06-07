import numpy as np
from PIL import Image as PILImage
import depthai
import cv2
import torch

from roboticstoolbox import DstarPlanner

from anytraverse.utils.cli.human_op.hoc_ctx import create_anytraverse_hoc_context
from anytraverse.utils.helpers import mask_poolers as mask_poolers
from anytraverse.utils.helpers.sensors.oakd import OakdCameraManager
from anytraverse.utils.helpers.grid_costmap import GridCostmap
from anytraverse.utils.pipelines.ws_human_op import AnyTraverseWebsocket
from threading import Thread
from anytraverse.utils.helpers.robots.unitree_zmq import UnitreeController
from anytraverse.utils.helpers import DEVICE


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


def show_costmap_with_path(
    costmap: np.ndarray, path: list[tuple[int, int]], wait: int = 0
):
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
    cv2.imshow("Costmap with Path", img)


def main():
    # Create AnyTraverse context
    anytraverse = create_anytraverse_hoc_context(
        init_prompts=[("floor", 1.0), ("mat", -0.5), ("shoes", -0.8)],
        mask_pooler=mask_poolers.ProbabilisticPooler,
    )

    # Start AnyTraverse WebSocket (for HOC UI, optional)
    ws_hoc = AnyTraverseWebsocket(anytraverse=anytraverse, port=7777)
    Thread(target=ws_hoc.start).start()

    # DepthAI camera setup
    depthai_pipeline = depthai.Pipeline()
    oakd = OakdCameraManager(pipeline=depthai_pipeline)

    # Costmap
    costmap = GridCostmap(x_bound=8, y_bound=5, resolution=0.15)

    # Robot controller (ZMQ)
    robot = UnitreeController(hostname="localhost", port=6969)
    robot.connect()

    # Start OAK-D device
    with depthai.Device(pipeline=depthai_pipeline) as depthai_device:
        oakd.setup_with_device(device=depthai_device)

        while True:
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
            temperature = 0.5
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
                costmap._height - int(0.20 / costmap._resolution),
                costmap._width // 2,
            )[::-1]

            print("Start point:", start_point)

            path, _ = d_star.query(start=start_point)
            show_costmap_with_path(costmap=costmap_grid, path=path.tolist())

            # if len(path) > int(1.0 / costmap._resolution):
            # Select initial motion direction
            w0, w1 = path[[0, int(0.7 / costmap._resolution)]]
            w0[1],w1[1] = costmap._height - w0[1], costmap._height - w1[1]
            print("Sending command to robot:", w0, w1)
            robot.send_command(start=w0, goal=w1)

            cv2.waitKey(1)


if __name__ == "__main__":
    main()
