import cv2
import os
import torch
import EasyPySpin
import time
import numpy as np

from config.utils import CameraConfig, HeightScoringConfig, PlaneFittingConfig
from utils.helpers.plane_fit import PCAPlaneFitter
from utils.models.clipseg.pooler import WeightedMaxPooler, ProbabilisticPooler

# Force OpenCV to use CPU
os.environ["OPENCV_DNN_BACKEND_CUDA"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Still allow CUDA for the HuggingFace model

from utils.pipelines.pipeline_002 import Pipeline2
from config.pipeline_002 import PipelineConfig
from PIL import Image

# Define the pipeline
# fx, fy, cx, cy = 2813.643275, 2808.326079, 969.285772, 624.049972
fx = 1172.04
fy = 1175.24
cx = 716.08
cy = 554.43

camera_params = {
    "camera_matrix": [
        [1172.0376188134887, 0.0, 716.0820787953663],
        [0.0, 1175.240980768119, 554.4305005206102],
        [0.0, 0.0, 1.0],
    ],
    "dist_coeffs": [
        [
            -0.2663530959471147,
            -1.082374877610441,
            0.005259570283547975,
            -0.003734076669586608,
            3.078586906738431,
        ]
    ],
}

# create camera matrix and distortion coefficients
camera_matrix = np.array(camera_params["camera_matrix"])
distortion_coefficients = np.array(camera_params["dist_coeffs"])

print("camera_matrix", camera_matrix)
print("distortion_coefficients", distortion_coefficients)

device = "cuda" if torch.cuda.is_available() else "mps"
print("device", device)

config = PipelineConfig(
    camera=CameraConfig(fx=fx, fy=fy, cx=cx, cy=cy),
    prompts=[("Human", 1), ("bush", -1)],
    device=device,
    height_scoring=HeightScoringConfig(alpha=30, z_thresh=0.1),
    plane_fitting=PlaneFittingConfig(
        fitter=PCAPlaneFitter(),
        trav_thresh=0.1,
    ),
    height_score=True,
    # mask_pooler=ProbabilisticPooler(),
    mask_pooler=WeightedMaxPooler(),
)

pipeline = Pipeline2(config=config)

# cv2 camera capture
cap = cv2.VideoCapture(0)

# Initialize the camera
# cap = EasyPySpin.VideoCapture(0)

#
# cap.set(cv2.CAP_PROP_EXPOSURE, 10000)  # us
# cap.set(cv2.CAP_PROP_GAIN, 10)  # dB

# Create window
cv2.namedWindow("Traversability", cv2.WINDOW_NORMAL)
prev_time = time.time()

while True:
    try:
        ret, frame = cap.read()
        if ret:
            # use camera parameters to correct image
            frame = cv2.undistort(frame, camera_matrix, distortion_coefficients, None)
            # resize frame
            frame = cv2.resize(frame, (640, 420))
            # frame = cv2.resize(frame, (1280, 720))

            # bgr to rgb
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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
            mask_binary = mask_np > 0.2
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
