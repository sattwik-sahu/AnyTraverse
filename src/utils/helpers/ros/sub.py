import roslibpy
from abc import ABC, abstractmethod
from typing import TypeVar

TRosMsg = TypeVar("TRosMsg", bound=dict)

class ROS_Subscriber[TRosMsg](ABC):
    """
    A ROS topic subscriber wrapper that reads messages from a given topic and processes them.

    Attributes:
        _client (roslibpy.Ros): A ROS client object.
        _subscriber (roslibpy.Topic): A ROS topic subscriber object.
    """

    _client: roslibpy.Ros
    _subscriber: roslibpy.Topic
    
    def __init__(self, host: str, port: int, topic: str, topic_dtype: str) -> None:
        """
        Initialize the ROS subscriber.

        Args:
            host (str): The ROS bridge websocket host.
            port (str): The ROS bridge websocket port.
            topic (str): The topic to subscribe to.
            topic_dtype (str): The topic data type.
        """
        self._client = roslibpy.Ros(host=host, port=port)
        self._subscriber = roslibpy.Topic(self._client, topic, topic_dtype)

    @abstractmethod
    def _process_msg(self, msg: TRosMsg) -> None:
        pass

    def spin(self) -> None:
        self._subscriber.subscribe(self._process_msg)
        self._client.run_forever()
