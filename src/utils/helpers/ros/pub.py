import rclpy
from rclpy.node import Node
from rclpy.subscription import Subscription
from sensor_msgs.msg import Image as ROS_Image, CompressedImage as ROS_CompressedImage
import cv2
from cv_bridge import CvBridge
from PIL import Image
from rich.console import Console
from typing import Type
from abc import ABC, abstractmethod


type TImage = ROS_Image | ROS_CompressedImage


class CameraSubscriber(Node, ABC):
    _cam_sub: Subscription
    _bridge: CvBridge
    _console: Console
    _topic_name: str

    def __init__(
        self,
        console: Console,
        topic_name: str,
        image_topic_type: Type[TImage],
        node_name: str = "anytraverse_cam_sub",
    ) -> None:
        super().__init__(node_name=node_name)
        self._console = console
        self._bridge = CvBridge()
        self._cam_sub = self.create_subscription(
            msg_type=image_topic_type,
            topic=topic_name,
            callback=self._cam_callback,
            qos_profile=10,
        )

    def _log(self, msg: str) -> None:
        self._console.log(f"[yellow]({self.get_name()})[/] {msg}")

    def _cam_callback(self, image: TImage) -> None:
        cv_img = None
        if isinstance(image, ROS_Image):
            cv_img = self._bridge.imgmsg_to_cv2(img_msg=image, desired_encoding="bgr8")
        else:
            cv_img = self._bridge.compressed_imgmsg_to_cv2(
                cmprs_img_msg=image, desired_encoding="bgr8"
            )
        # cv2.imshow(f"Frame: {self.get_name()}", cv_img)
        # cv2.waitKey(1)
        # self._image = Image.fromarray(cv_img)
        self.callback(image=Image.fromarray(cv_img))

    @abstractmethod
    def callback(self, image: Image.Image):
        pass


# def main(args=None):
#     console = Console()
#     rclpy.init(args=args)
#     cam_subscriber = CameraSubscriber(
#         console=console,
#         topic_name="/oak/rgb/image_raw/compressed",
#         image_topic_type=ROS_CompressedImage,
#     )
#     rclpy.spin(node=cam_subscriber)
#     cam_subscriber.destroy_node()
#     rclpy.shutdown()


# if __name__ == "__main__":
#     main()
