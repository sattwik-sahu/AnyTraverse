import json
from typing import TypedDict

import numpy as np
from numpy import typing as npt
import websockets


class Command(TypedDict):
    start: tuple[int, int]
    target: tuple[int, int]


class UnitreeController:
    def __init__(self, hostname: str = "localhost", port: int = 6969) -> None:
        self._hostname = hostname
        self._port = port
        self._ws: websockets.ClientConnection | None = None

    async def connect(self) -> None:
        self._ws = await websockets.connect(f"ws://{self._hostname}:{self._port}")
        print(f"Connected to Unitree controller at ws://{self._hostname}:{self._port}")
        # await asyncio.sleep(5)

    async def send_command(
        self, start: npt.NDArray[np.int16], goal: npt.NDArray[np.int16]
    ) -> None:
        if self._ws is None:
            raise RuntimeError("WebSocket connection is not established.")

        command: Command = {
            "start": tuple(start.tolist()),
            "target": tuple(goal.tolist()),
        }
        await self._ws.send(json.dumps(command))
        print(f"Sent command: {command}")
