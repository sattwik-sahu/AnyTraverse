from anytraverse.config.datasets import DatasetConfig, RGBValue
from dataclasses import dataclass


@dataclass
class Classes:
    VOID = (0, 0, 0)
    DIRT = (108, 64, 20)
    SAND = (255, 229, 204)
    GRASS = (0, 102, 0)
    TREE = (0, 255, 0)
    POLE = (0, 153, 153)
    WATER = (0, 128, 255)
    SKY = (0, 0, 255)
    VEHICLE = (255, 255, 0)
    OBJECT = (255, 0, 127)
    ASPHALT = (64, 64, 64)
    GRAVEL = (255, 128, 0)
    BUILDING = (255, 0, 0)
    MULCH = (153, 76, 0)
    ROCK_BED = (102, 102, 0)
    LOG = (102, 0, 0)
    BICYCLE = (0, 255, 128)
    PERSON = (204, 153, 255)
    FENCE = (102, 0, 204)
    BUSH = (255, 153, 204)
    SIGN = (0, 102, 102)
    ROCK = (153, 204, 255)
    BRIDGE = (102, 255, 255)
    CONCRETE = (101, 101, 11)
    PICNIC_TABLE = (114, 85, 47)


config = DatasetConfig[RGBValue](
    name="RUGD",
    traversibles=[
        Classes.GRASS,
        Classes.DIRT,
        Classes.ASPHALT,
        Classes.CONCRETE,
        Classes.SAND,
        Classes.GRAVEL,
        Classes.MULCH,
        Classes.ROCK_BED,
    ],
)
