from anytraverse.config.datasets import DatasetConfig, RGBValue
from dataclasses import dataclass


@dataclass
class Classes:
    SKY = (0, 120, 255)
    GRASS = (0, 255, 0)
    TRAIL = (170, 170, 170)
    VEGETATION = (102, 102, 51)


config = DatasetConfig[RGBValue](
    name="RUGD",
    traversibles=[
        Classes.GRASS,
        Classes.TRAIL,
    ],
)
