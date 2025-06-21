import os
import cv2
import time
import numpy as np
import torch
from PIL import Image
from typing import Optional, TypedDict
import pandas as pd
from datetime import datetime as Datetime
from anytraverse.config.utils import WeightedPrompt
import json


class FrameData(TypedDict):
    trav_roi: float
    unc_roi: float
    prompts: float
    timestamp: Datetime


class AnyTraverseLogger:
    """
    A class for logging RGB images, traversability maps, and uncertainty maps as a video.

    Attributes:
        save_dir (str): Directory to save the video file.
        video_path (str): Full path of the video file.
        writer (cv2.VideoWriter): OpenCV video writer object.
        fps (int): Frames per second of the saved video.
        frame_size (tuple[int, int]): Frame size (width, height) of the concatenated image.
    """

    def __init__(self, save_dir: str, fps: int = 10) -> None:
        """
        Initializes the TraversabilityVideoLogger.

        Args:
            save_dir (str): Directory where the video file will be saved.
            fps (int, optional): Frames per second of the output video. Defaults to 10.
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

        # Generate a filename based on current timestamp
        timestamp = time.strftime("%Y-%m-%d_%H-%M")
        self.video_path = os.path.join(save_dir, f"{timestamp}.avi")
        self.log_path = os.path.join(save_dir, f"{timestamp}.json")
        self.log_df = pd.DataFrame(
            columns=["trav_roi", "unc_roi", "prompts", "timestamp"]
        )

        self.writer: cv2.VideoWriter | None = None
        self.fps = fps
        self.frame_size: Optional[tuple[int, int]] = None

    def _init_writer(self, height: int, width: int) -> None:
        """
        Initializes the OpenCV VideoWriter with the given dimensions.

        Args:
            height (int): Height of the frame.
            width (int): Width of the frame.
        """
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.writer = cv2.VideoWriter(
            self.video_path, fourcc, self.fps, (width, height)
        )

    def add_frame(
        self, image: Image.Image, trav_map: torch.Tensor, unc_map: torch.Tensor
    ) -> np.ndarray:
        """
        Adds a frame to the video by combining the image with traversability and uncertainty maps.

        Args:
            image (PIL.Image.Image): The RGB input image.
            trav_map (torch.Tensor): Traversability map tensor of shape (H, W), values in [0, 1].
            unc_map (torch.Tensor): Uncertainty map tensor of shape (H, W), values in [0, 1].

        Returns:
            np.ndarray: The horizontally concatenated frame that was written to the video.
        """
        # Convert PIL Image to BGR OpenCV image
        image_cv = np.array(image)

        # Normalize and convert traversability and uncertainty maps to 3-channel BGR images
        def to_heatmap(tensor: torch.Tensor) -> np.ndarray:
            """
            Converts a single-channel [0,1] torch tensor to a BGR heatmap.

            Args:
                tensor (torch.Tensor): Input tensor.

            Returns:
                np.ndarray: BGR heatmap image.
            """
            arr = (tensor.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(arr, cv2.COLORMAP_PLASMA)
            return heatmap

        trav_map_cv = to_heatmap(trav_map)
        unc_map_cv = to_heatmap(unc_map)

        # Concatenate the three images horizontally
        combined = np.concatenate((image_cv, trav_map_cv, unc_map_cv), axis=1)

        # Initialize video writer if not already initialized
        if self.writer is None:
            height, width, _ = combined.shape
            self.frame_size = (width, height)
            self._init_writer(height, width)

        # Write frame to video
        self.writer.write(combined)

        return combined

    def add_data(
        self, trav_roi: float, unc_roi: float, prompts: list[WeightedPrompt]
    ) -> None:
        """
        Adds metadata for the current frame to the log.

        Args:
            trav_roi (float): Traversability ROI value.
            unc_roi (float): Uncertainty ROI value.
            prompts (float): Prompts value.
        """
        timestamp = Datetime.now()
        self.log_df.loc[self.log_df.shape[0]] = {  # type: ignore
            "trav_roi": trav_roi,
            "unc_roi": unc_roi,
            "prompts": prompts,
            "timestamp": timestamp,
        }

    def close(self) -> None:
        """
        Closes the video writer and finalizes the video file.
        """
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        self.log_df.to_json(self.log_path, orient="records", date_format="iso")
