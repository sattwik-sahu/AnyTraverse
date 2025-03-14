import base64
import numpy as np
import torch
import cv2 as cv
from PIL import Image
import io
from utils.helpers.ros.sub import ROS_Subscriber
from abc import ABC, abstractmethod
from typing_extensions import override
from utils.cli.human_op.models import CompressedImage


class ImageSubscriber(ROS_Subscriber[CompressedImage], ABC):
    """
    A ROS topic subscriber wrapper that reads images from a given topic and processes them.

    Attributes:
        _client (roslibpy.Ros): A ROS client object.
        _subscriber (roslibpy.Topic): A ROS topic subscriber object.
    """

    def __init__(self, host: str, port: int, topic: str) -> None:
        """
        Initialize the image subscriber.

        Args:
            host (str): The ROS bridge websocket host.
            port (str): The ROS bridge websocket port.
            topic (str): The topic to subscribe to.
        """
        topic_dtype = "sensor_msgs/CompressedImage"
        super().__init__(host, port, topic, topic_dtype)

    @abstractmethod
    def _process_image(self, img: Image.Image) -> None:
        """
        Process the received image.

        Args:
            img (Image.Image): The received image.
        """
        pass

    @override
    def _process_msg(self, msg: CompressedImage) -> None:
        """
        Process the received image message.

        Args:
            msg (dict): The received image message.
        """
        base64_bytes = msg["data"].encode("ascii")
        image_bytes = io.BytesIO(base64.b64decode(base64_bytes))
        img = Image.open(image_bytes)
        img = img.convert("RGB")  # Ensure image is in RGB format
        self._process_image(img)


class ROS_HumanOperatorContext(ImageSubscriber):
    """
    A ROS topic subscriber wrapper that reads images from a given topic and processes them.

    Attributes:
        _client (roslibpy.Ros): A ROS client object.
        _subscriber (roslibpy.Topic): A ROS topic subscriber object.
    """
    _hoc_ctx: 

    def __init__(self, host: str, port: int, topic: str) -> None:
        """
        Initialize the image subscriber.

        Args:
            host (str): The ROS bridge websocket host.
            port (str): The ROS bridge websocket port.
            topic (str): The topic to subscribe to.
        """
        super().__init__(host, port, topic)

    @override
    def _process_image(self, img: Image.Image) -> None:
        """
        Process the received image.

        Args:
            img (Image.Image): The received image.
        """
        img = np.array(img)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        cv.imshow("Human Operator Context", img)
        cv.waitKey(1)


if __name__ == "__main__":
    host = "0.0.0.0"
    port = 6969
    topic = "/camera/rgb"
    human_op_context = ROS_HumanOperatorContext(host, port, topic)
    human_op_context.spin()
