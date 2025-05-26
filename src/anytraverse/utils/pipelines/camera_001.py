import cv2
import os
import torch
import EasyPySpin
import time
import numpy as np

from anytraverse.config.utils import (
    CameraConfig,
    HeightScoringConfig,
    PlaneFittingConfig,
)
from anytraverse.utils.helpers.plane_fit import PCAPlaneFitter
from anytraverse.utils.helpers.pooler import (
    WeightedMaxPooler,
    ProbabilisticPooler,
)

# Force OpenCV to use CPU
os.environ["OPENCV_DNN_BACKEND_CUDA"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Still allow CUDA for the HuggingFace model

from anytraverse.utils.pipelines.base import Pipeline2
from anytraverse.config.pipeline_002 import PipelineConfig
from PIL import Image

# Define the pipeline
fx, fy, cx, cy = 2813.643275, 2808.326079, 969.285772, 624.049972
device = "cuda" if torch.cuda.is_available() else "mps"
print("device", device)

config = PipelineConfig(
    camera=CameraConfig(fx=fx, fy=fy, cx=cx, cy=cy),
    prompts=[("Floor", 1.0), ("Chair", -1.0)],
    device=device,
    height_scoring=HeightScoringConfig(alpha=30, z_thresh=0.1),
    plane_fitting=PlaneFittingConfig(
        fitter=PCAPlaneFitter(),
        trav_thresh=0.1,
    ),
    height_score=False,
    mask_pooler=ProbabilisticPooler(),
    # mask_pooler=WeightedMaxPooler(),
)

pipeline = Pipeline2(config=config)

# Initialize the camera
cap = EasyPySpin.VideoCapture(0)
width = cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
height = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

# Create window
cv2.namedWindow("Traversability", cv2.WINDOW_NORMAL)
prev_time = time.time()

while True:
    try:
        ret, frame = cap.read()
        if ret:
            # resize frame
            frame = cv2.resize(frame, (640, 420))

            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            # frame dtype reducded to float16
            # frame = frame.astype(np.int8)

            image = Image.fromarray(frame)
            trav_mask = pipeline(image=image)

            # Convert mask to color overlay
            mask_np = trav_mask.cpu().numpy()
            # mask_colored = cv2.applyColorMap((mask_np * 255).astype(np.uint8), cv2.COLORMAP_JET)
            # mask_colored = cv2.cvtColor(mask_colored, cv2.COLOR_BGR2RGB)
            mask_binary = mask_np > 0.5
            mask_colored = cv2.applyColorMap(
                (mask_binary * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS
            )
            # binary segment the mask
            # mask = mask_np > 0.

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Combine original frame with mask
            # convert mas
            overlay = cv2.addWeighted(frame, 1, mask_colored, 0.3, 0)

            # Add FPS text to frame
            cv2.putText(
                overlay,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # Show combined image
            cv2.imshow("Traversability", overlay)

            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
