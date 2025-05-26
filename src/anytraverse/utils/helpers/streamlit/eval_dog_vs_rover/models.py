from enum import Enum
from pathlib import Path
from typing import List, TypedDict, TypeVar

from pydantic import BaseModel

from anytraverse.utils.pipelines.base import WeightedPrompt

WeightedPrompts = List[WeightedPrompt]
TVehicleData = TypeVar("TVehicleData")


class VehicleConfig(BaseModel):
    prompts: WeightedPrompts
    perform_height_scoring: bool


class VehiclesModel[TVehicleData](BaseModel):
    dog: TVehicleData
    rover: TVehicleData


class ExampleType(str, Enum):
    SUCCESS = "Success"
    FAILURE = "Failure"
    BAD_GT = "Bad Ground Truth"


class ExampleMetadataModel(BaseModel):
    type: ExampleType
    remarks: str = ""


class ExampleDataModel(BaseModel):
    rgb: Path
    gt_dog: Path
    gt_rover: Path
    vehicles: VehiclesModel[VehicleConfig]
    iou: VehiclesModel[float]
    metadata: ExampleMetadataModel


class LogsModel(BaseModel):
    csv: Path
    logs: List[ExampleDataModel]


class Vehicles[TVehicleData](TypedDict):
    dog: TVehicleData
    rover: TVehicleData
