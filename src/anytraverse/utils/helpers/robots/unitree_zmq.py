import json
import zmq
from typing import TypedDict
import numpy as np
from numpy import typing as npt


class Command(TypedDict):
    start: tuple[int, int]
    target: tuple[int, int]


class UnitreeController:
    def __init__(self, hostname: str = "localhost", port: int = 6969) -> None:
        self._hostname = hostname
        self._port = port
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)

    def connect(self) -> None:
        self._socket.connect(f"tcp://{self._hostname}:{self._port}")
        print(f"Connected to Unitree controller at tcp://{self._hostname}:{self._port}")

    def send_command(
        self, start: npt.NDArray[np.int16], goal: npt.NDArray[np.int16]
    ) -> None:
        command: Command = {
            "start": tuple(start.tolist()),
            "target": tuple(goal.tolist()),
        }
        self._socket.send_json(command)
        # response = self._socket.recv_json()
        # response["yaw_speed"] = np.round(np.rad2deg(response["yaw_speed"]), decimals=3)
        # print(f"Sent command: {command}")
        # return response

    def stop_robot(self) -> None:
        self.send_command(
            start=np.zeros(2, dtype=np.int16), goal=np.zeros(1, dtype=np.int16)
        )
