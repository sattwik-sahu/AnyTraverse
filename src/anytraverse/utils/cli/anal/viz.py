from pathlib import Path
from typing import Iterator, Tuple

import cv2
import numpy as np
from numpy import typing as npt
from PIL import Image

from anytraverse.utils.helpers.video import extract_frames_from_video
from anytraverse.utils.pipelines.base import Pipeline2

from rich.progress import Progress


def run(
    pipeline: Pipeline2,
    video_path: Path,
    output_path: Path,
    color: Tuple[int, int, int] = (255, 255, 255),
    threshold: float = 0.5,
    overlay_alpha: float = 0.3,
    nth_frame: int = 1,
):
    frames: Iterator[npt.NDArray[np.uint8]] = extract_frames_from_video(
        path=video_path, nth_frame=nth_frame
    )

    with Progress() as progress:
        task = progress.add_task(description="Processing frames", total=None)

        for i, frame in enumerate(frames):
            # Create image object
            image: Image.Image = Image.fromarray(frame)

            # Run the pipeline
            output = pipeline(image=image).cpu().numpy()

            # Threshold the image
            mask = output > threshold

            # Overlay color (on GPU)
            color_arr = np.array(color)[::-1].reshape(1, 1, 3)
            mask_overlayed = frame.copy()
            mask_overlayed[mask] = np.round(
                (1 - overlay_alpha) * mask_overlayed[mask] + overlay_alpha * color_arr
            ).astype(dtype=np.uint8)

            # Write output to path
            output_file = output_path / Path(f"frame_{str(i * nth_frame).zfill(6)}.jpg")
            cv2.imwrite(output_file.as_posix(), img=mask_overlayed)

            progress.update(task_id=task)
