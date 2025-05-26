from anytraverse.config.datasets import DatasetConfig
from dataclasses import dataclass
from anytraverse.config.pipeline_002 import CameraConfig


fx, fy, cx, cy = 2813.643275, 2808.326079, 969.285772, 624.049972
camera_config = CameraConfig(fx=fx, fy=fy, cx=cx, cy=cy)


@dataclass
class Classes:
    DIRT = 1
    GRASS = 3
    TREE = 4
    POLE = 5
    WATER = 6
    SKY = 7
    VEHICLE = 8
    OBJECT = 9
    ASPHALT = 10
    BUILDING = 12
    LOG = 15
    PERSON = 17
    FENCE = 18
    BUSH = 19
    CONCRETE = 23
    BARRIER = 27
    PUDDLE = 31
    MUD = 33
    RUBBLE = 34


config = DatasetConfig[int](
    name="RELLIS-3D",
    traversibles=[
        Classes.GRASS,
        Classes.DIRT,
        Classes.ASPHALT,
        Classes.CONCRETE,
        # Classes.PUDDLE,
        Classes.MUD,
    ],
)
