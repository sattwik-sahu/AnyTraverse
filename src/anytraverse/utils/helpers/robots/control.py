from typing import TypedDict


class RobotWaypointCommand(TypedDict):
    start: tuple[int, int]
    target: tuple[int, int]


class RobotControlCommand(TypedDict):
    velocity: list[float]
    yaw_speed: float
