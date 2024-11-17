from dataclasses import dataclass
from typing import List, Tuple

RGBValue = Tuple[int, int, int]


@dataclass
class DatasetConfig[T_Traversible: int | RGBValue]:
    name: str
    traversibles: List[T_Traversible]
