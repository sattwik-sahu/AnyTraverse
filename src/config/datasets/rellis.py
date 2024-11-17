from config.datasets import DatasetConfig
from dataclasses import dataclass


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
    ]
)
