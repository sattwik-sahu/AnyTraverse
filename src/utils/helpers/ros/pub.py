import roslibpy
from typing import TypeVar


TRosMsg = TypeVar("TRosMsg", bound=dict)


class ROS_Publisher[TRosMsg]:
    """
    A ROS topic publisher wrapper that publishes messages to a given topic.

    Attributes:
        _client (roslibpy.Ros): A ROS client object.
        _publisher (roslibpy.Topic): A ROS topic publisher object.
    """

    _client: roslibpy.Ros
    _publisher: roslibpy.Topic

    def __init__(self, host: str, port: str, topic: str, topic_dtype: str) -> None:
        """
        Initialize the ROS publisher.

        Args:
            host (str): The ROS bridge websocket host.
            port (str): The ROS bridge websocket port.
            topic (str): The topic to publish to.
            topic_dtype (str): The topic data type.
        """
        self._client = roslibpy.Ros(host=host, port=port)
        self._publisher = roslibpy.Topic(self._client, topic, topic_dtype)
        self._publisher.advertise()

    def publish(self, msg: TRosMsg) -> None:
        self._publisher.publish(roslibpy.Message(values=msg))
