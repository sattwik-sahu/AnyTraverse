from pathlib import Path
import cv2
from typing import Iterator
import numpy as np
from numpy import typing as npt


def extract_frames_from_video(
    path: Path, nth_frame: int = 1
) -> Iterator[npt.NDArray[np.uint8]]:
    if not path.exists():
        raise FileNotFoundError(f"File {path} does not exist.")

    # Create video object
    video = cv2.VideoCapture(path.absolute().as_posix())

    # Counter variable
    n: int = 0

    # Frame extracted?
    success: bool = True

    # Run video frame extraction loop
    while success:
        success_, frame = video.read()
        success = success & success_
        if n % nth_frame == 0:
            # rgb_frame = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB).astype(np.uint8)
            rgb_frame = frame.astype(np.uint8)
            yield rgb_frame
        n += 1
