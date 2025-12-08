from enum import Enum, auto
import platform


class PlatformType(Enum):
    """
    Enumerator for the platform type.
    """

    WINDOWS = auto()
    """Windows"""

    MAC = auto()
    """MacOS"""

    LINUX = auto()
    """Linux"""


PLATFORM: PlatformType | None = None
"""The current platform (Windows, Mac or Linux)"""


match platform.system():
    case "Windows":
        PLATFORM = PlatformType.WINDOWS
    case "Darwin":
        PLATFORM = PlatformType.MAC
    case "Linux":
        PLATFORM = PlatformType.LINUX
    case _:
        PLATFORM = None
