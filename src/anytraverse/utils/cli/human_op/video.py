from pathlib import Path

from anytraverse.utils.cli.human_op.models import DatasetVideo, DatasetVideosConfigModel


def get_video_path(dataset: DatasetVideo) -> Path:
    """
    Gets the path for a particular dataset video using the `DatasetVideo`
    enumerator.

    Args:
        dataset (DatasetVideo): The enumerator

    Returns:
        Path: The `Path` object pointing to the video corresponding to the
            `dataset` enumerator value.
    """
    # Open the videos.json file
    with open("data/videos/list.json", "r") as f:
        videos = DatasetVideosConfigModel.model_validate_json(f.read()).videos
    return videos[dataset]
