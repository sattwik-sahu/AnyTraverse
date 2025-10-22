from abc import ABC, abstractmethod
from anytraverse.utils.helpers.robots.control import RobotControlCommand


class ControlPublisher(ABC):
    def __init__(*args, **kwargs) -> None:
        pass

    @abstractmethod
    def send(self, command: RobotControlCommand) -> None:
        pass
