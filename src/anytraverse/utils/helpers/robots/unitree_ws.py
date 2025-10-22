import websockets
import json
from typing import TypedDict
from anytraverse.utils.helpers.robots.unitree_go1 import RobotController


type CommandVel = float | list[float]


class ControlCommand(TypedDict):
    velocity: CommandVel
    yaw_speed: float


class UnitreeWebsocketServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 6969) -> None:
        self._robot = RobotController()

    def _convert_to_unitree_command(self, control: ControlCommand) -> ControlCommand:
        if type(control["velocity"]) is float:
            control["velocity"] = [control["velocity"], 0.0]
        return control

    def send_control(self, velocity: CommandVel, yaw_speed: float) -> None:
        pass

    def connect(self) -> None:
        pass
