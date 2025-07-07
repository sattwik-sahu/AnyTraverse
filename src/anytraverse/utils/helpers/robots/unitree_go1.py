import math
import time
import sys
from typing import Tuple, List

import numpy as np

# Add your robot SDK path
sys.path.append("../lib/python/amd64")
import robot_interface as sdk  # type: ignore


class CommandBuilder:
    """Builds high-level walking commands for the robot."""

    MAX_LINEAR_VEL = 0.4
    YAW_GAIN = 2.0
    FORWARD_GAIN = 0.8
    TURN_SLOWDOWN_THRESHOLD = math.pi / 4
    TURN_SLOWDOWN_FACTOR = 0.3

    @staticmethod
    def compute_velocity(
        start: Tuple[float, float], target: Tuple[float, float]
    ) -> Tuple[List[float], float]:
        """
        Computes velocity and yaw speed to reach the target point from current location.

        Args:
            start: (x, y) current position in grid or world coordinates.
            target: (x, y) target position in grid or world coordinates.

        Returns:
            velocity: [forward, lateral] velocity vector.
            yaw_speed: yaw angular velocity.
        """
        dx = target[0] - start[0]
        dy = target[1] - start[1]
        target_yaw = math.atan2(dx, dy)

        # Normalize yaw to [-pi, pi]
        target_yaw = (target_yaw + math.pi) % (2 * math.pi) - math.pi

        # Forward motion along y-axis (relative to robot frame)
        forward_vel = CommandBuilder.FORWARD_GAIN * dy
        forward_vel = max(
            -CommandBuilder.MAX_LINEAR_VEL,
            min(CommandBuilder.MAX_LINEAR_VEL, forward_vel),
        )

        # Reduce speed if turning sharply
        if abs(target_yaw) > CommandBuilder.TURN_SLOWDOWN_THRESHOLD:
            forward_vel *= CommandBuilder.TURN_SLOWDOWN_FACTOR

        return [forward_vel, 0.0], CommandBuilder.YAW_GAIN * target_yaw


class RobotController:
    """Controls the robot via UDP commands."""

    def __init__(
        self,
        robot_ip: str = "192.168.123.161",
        local_port: int = 8080,
        robot_port: int = 8082,
    ):
        self.udp = sdk.UDP(0xEE, local_port, robot_ip, robot_port)
        self.cmd = sdk.HighCmd()
        self.state = sdk.HighState()
        self.udp.InitCmdData(self.cmd)

    def send_command(
        self,
        velocity: List[float],
        yaw_speed: float,
        gait: int = 1,
        mode: int = 2,
        body_height: float = 0.0,
        foot_raise: float = 0.05,
        delay: float = 1.0,
    ) -> None:
        """
        Sends a movement command to the robot.

        Args:
            velocity: [x, y] velocity.
            yaw_speed: yaw angular velocity.
            gait: gait type.
            mode: control mode.
            body_height: body height offset.
            foot_raise: foot raise height.
            delay: optional delay before sending.
        """
        self.cmd.mode = mode
        self.cmd.gaitType = gait
        self.cmd.velocity = velocity
        self.cmd.yawSpeed = yaw_speed
        self.cmd.bodyHeight = body_height
        self.cmd.footRaiseHeight = foot_raise

        time.sleep(delay)
        self.udp.SetSend(self.cmd)
        self.udp.Send()


# def main():
#     # Grid setup
#     grid_height = 50
#     grid_width = int(8 / 0.15)
#     start_pos = (grid_width / 2, 0)

#     controller = RobotController()
#     target_pos = (grid_width / 2 + 5, 9)  # Example target

#     velocity, yaw_speed = CommandBuilder.compute_velocity(
#         start=start_pos, target=target_pos
#     )
#     controller.send_command(velocity=velocity, yaw_speed=yaw_speed)


# if __name__ == "__main__":
#     main()
