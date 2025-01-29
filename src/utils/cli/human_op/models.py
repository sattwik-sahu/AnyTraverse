from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, TypedDict

from pydantic import BaseModel

from config.utils import WeightedPrompt

# from sqlmodel import SQLModel, create_engine, MetaData, Field


class DatasetVideo(Enum):
    OURS_FOREST = "AnyTraverse Forest Nursery"
    OURS_HILLSIDE = "AnyTraverse IISERB Hillside"
    RELLIS1 = "RELLIS Episode 0"
    RELLIS2 = "RELLIS Episode 3"
    RUGD_TRAIL = "RUGD Hill"
    RUGD_CREEK = "RUGD Creek"
    FREIBURG_FOREST = "Freiburg Forest"


class DatasetVideosConfigModel(BaseModel):
    videos: Dict[DatasetVideo, Path]


class DriveStatus(Enum):
    OK = "ok"
    BAD_ROI = "bad_roi"
    UNSEEN_SCENE = "unseen_scene"


class Thresholds(TypedDict):
    ref_sim: float
    roi: float
    seg: float


class HumanOperatorCallLog(BaseModel):
    # id: Optional[int] = Field(default=None, primary_key=True)
    video: DatasetVideo
    frame_inx: int
    ref_sim_score: float
    trav_roi: float
    thresh: Thresholds
    human_op_called: bool
    human_op_call_req: bool
    human_op_call_type: DriveStatus | None = None
    prompts: List[WeightedPrompt]
    hist_used_succ: bool

class HumanOperatorCallLogs(BaseModel):
    logs: List[HumanOperatorCallLog]


class LoopbackLogModel(BaseModel):
    video: DatasetVideo
    roi_calls: List[int]
    scene_change_calls: List[int]


class LoopbackLogsModel(BaseModel):
    logs: List[LoopbackLogModel]

# log1 = HumanOperatorCallLog(
#     video=DatasetVideo.RUGD_CREEK,
#     ref_sim_score=0.4,
#     frame_inx=9,
#     trav_roi=0.34,
#     thresh=Thresholds(ref_sim=0.9, roi=0.5, seg=0.25),
#     human_op_called=True,
#     human_op_call_req=True,
#     human_op_call_type=DriveStatus.BAD_ROI,
# )

# Create the engine
# engine = create_engine("sqlite:///data/logs/human-op.db")

# if __name__ == "__main__":
#     # Create database and tables
#     SQLModel.metadata.create_all(engine)
