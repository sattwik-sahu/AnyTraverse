import zmq
from abc import ABC, abstractmethod
from typing_extensions import override
from typing import TypedDict
import json
import time


class RobotWaypointCommand(TypedDict):
    start: tuple[int, int]
    target: tuple[int, int]


class RobotControlCommand(TypedDict):
    velocity: list[float]
    yaw_speed: float


RobotCommand = RobotControlCommand | RobotWaypointCommand


class ZMQPublisher[TMessage](ABC):
    """
    A class-based ZeroMQ Publisher using PUB socket.

    Attributes:
        address (str): The IPC or TCP address to bind to.
        context (zmq.Context): The ZeroMQ context.
        socket (zmq.Socket): The PUB socket.
    """

    def __init__(self, address: str) -> None:
        """
        Initializes the ZMQPublisher.

        Args:
            address (str): Address to bind the publisher to.
        """
        self.address = "tcp://*:5555"
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 10)  # Optional: High-water mark for queuing
        self.socket.bind(self.address)
        time.sleep(0.5)
        

    @abstractmethod
    def serialize_message(self, message: TMessage) -> str:
        pass

    def send(self, topic: str, message: TMessage) -> None:
        """
        Sends a message with a topic prefix.

        Args:
            topic (str): Topic string used by subscribers to filter.
            message (str): The message to send.
        """
        serialized_message = self.serialize_message(message=message)
        full_msg = f"{topic} {serialized_message}"
        print(f"[SOCKET] > {full_msg}")
        self.socket.send_string(full_msg)

    def close(self) -> None:
        """Closes the socket and terminates the context."""
        self.socket.close()
        self.context.term()


class UnitreeZMQPublisher(ZMQPublisher[RobotCommand]):
    def __init__(self) -> None:
        super().__init__(address="anytraverse")

    @override
    def serialize_message(self, message: RobotCommand) -> str:
        return json.dumps(message)
