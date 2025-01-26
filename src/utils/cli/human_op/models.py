from pydantic import BaseModel
from pathlib import Path
from enum import Enum
from typing import Dict


class DatasetVideo(Enum):
    OURS_FOREST = "AnyTraverse Forest Nursery"
    OURS_HILLSIDE = "AnyTraverse IISERB Hillside"
    RELLIS1 = "RELLIS Episode 0"
    RELLIS2 = "RELLIS Episode 3"
    RUGD1 = "RUGD Hill"
    RUGD2 = "RUGD Creek"


class DatasetVideosConfigModel(BaseModel):
    videos: Dict[DatasetVideo, Path]


data = DatasetVideosConfigModel(
    videos={
        DatasetVideo.OURS_HILLSIDE: Path(
            "/mnt/toshiba_hdd/datasets/iiserb/anytraverse/2024-12-10__hound_hillside/video_012.avi"
        ),
        DatasetVideo.OURS_FOREST: Path(
            "/mnt/toshiba_hdd/datasets/iiserb/anytraverse/2024-12-10__hound_hillside/video_012.avi"
        ),
        DatasetVideo.RELLIS1: Path(
            "/mnt/toshiba_hdd/datasets/rellis-3d/Rellis-3D-images/00000/pylon_camera_node/video.mp4"
        ),
        DatasetVideo.RELLIS2: Path(
            "/mnt/toshiba_hdd/datasets/rellis-3d/Rellis-3D-images/00000/pylon_camera_node/video.mp4"
        ),
        DatasetVideo.RUGD1: Path(
            "/mnt/toshiba_hdd/datasets/iiserb/anytraverse/2024-12-10__hound_hillside/video_012.avi"
        ),
        DatasetVideo.RUGD2: Path(
            "/mnt/toshiba_hdd/datasets/iiserb/anytraverse/2024-12-10__hound_hillside/video_012.avi"
        ),
    }
)

with open("data/videos/test.json", "w") as f:
    f.write(data.model_dump_json())
