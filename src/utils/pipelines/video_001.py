import cv2
import os
import torch
import EasyPySpin
import time
import numpy as np
import argparse
from pathlib import Path

from tqdm import tqdm

from config.utils import CameraConfig, HeightScoringConfig, PlaneFittingConfig
from utils.helpers.plane_fit import PCAPlaneFitter
from utils.models.clipseg.pooler import WeightedMaxPooler, ProbabilisticPooler

# Force OpenCV to use CPU
os.environ["OPENCV_DNN_BACKEND_CUDA"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Still allow CUDA for the HuggingFace model

from utils.pipelines.pipeline_002 import Pipeline2
from config.pipeline_002 import PipelineConfig
from PIL import Image

# Add argument parsing
parser = argparse.ArgumentParser(
    description="Process video file for traversability analysis"
)
parser.add_argument("video_path", type=str, help="Path to input video file")
parser.add_argument(
    "--output_dir",
    type=str,
    default="output",
    help="Output directory for frames and video",
)
args = parser.parse_args()

# Create output directory
output_dir = Path(args.output_dir)

# Create output directory structure
video_basename = os.path.basename(args.video_path).split(".")[0]
video_output_dir = output_dir / video_basename
frames_dir = video_output_dir / "frames"
video_output_dir.mkdir(exist_ok=True, parents=True)
frames_dir.mkdir(exist_ok=True)

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

device = "cuda" if torch.cuda.is_available() else "mps"
print("device", device)

config = PipelineConfig(
    camera=CameraConfig(fx=fx, fy=fy, cx=cx, cy=cy),
    prompts=[("Dirt", 1), ("dry grass", 0.9), ("bush", -1)],
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

# Setup video capture and writer
cap = cv2.VideoCapture(args.video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_video_path = str(video_output_dir / "output.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_count = 0
prev_time = time.time()

i = 0
step = 20

cv2.namedWindow("Traversability", cv2.WINDOW_NORMAL)
# Process frames with progress bar
for frame_count in tqdm(range(total_frames), desc="Processing frames"):
    ret, frame = cap.read()
    if not ret:
        break

    # Skip frames that are not at step intervals
    if frame_count % step != 0:
        continue

    # use camera matrix and distortion coefficients
    frame = cv2.undistort(frame, camera_matrix, distortion_coefficients, None)

    # resize frame
    frame = cv2.resize(frame, (640, 420))
    # frame = cv2.resize(frame, (1280, 720))

    # bgr to rgb
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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
    # mask_binary = mask_np > 0.2
    # mask_colored = cv2.applyColorMap(
    #     (mask_binary * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS
    # )
    # binary segment the mask
    # mask = mask_np > 0.

    # Create binary mask
    mask_binary = mask_np > 0.3

    # Create blank colored mask
    h, w = mask_binary.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    # Apply colormap only to threshold pixels
    colored_mask[mask_binary] = cv2.applyColorMap(
        np.full((np.sum(mask_binary), 1), 255, dtype=np.uint8), cv2.COLORMAP_VIRIDIS
    )[0]

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Overlay colored mask on original frame
    overlay = cv2.addWeighted(frame, 1, colored_mask, 0.3, 0)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Combine original frame with mask
    # convert mas
    # overlay = cv2.addWeighted(frame, 1, mask_colored, 0.3, 0)

    # Add FPS text to frame
    # cv2.putText(overlay, f'FPS: {fps:.1f}', (10, 30),
    #            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Traversability", overlay)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Save frame
    frame_path = frames_dir / f"frame_{frame_count:04d}.png"

    # if i%10 == 0:
    cv2.imwrite(str(frame_path), overlay)

    # Write to video
    # out.write(overlay)

    frame_count += 1

    if frame_count % step == 0:
        tqdm.write(f"Processing frame {frame_count}/{total_frames}")

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed {frame_count + 1} frames")
print(f"Output video saved to: {output_video_path}")
print(f"Output frames saved to: {frames_dir}")
