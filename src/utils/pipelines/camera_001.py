import cv2
import matplotlib.pyplot as plt
import os

# Force OpenCV to use CPU
os.environ["OPENCV_DNN_BACKEND_CUDA"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Still allow CUDA for the HuggingFace model

from utils.pipelines.pipeline_002 import Pipeline2
from config.pipeline_002 import PipelineConfig
from PIL import Image

# Initialize the pipeline
pipeline = Pipeline2(
    config=PipelineConfig(
        prompts=[
            ("guy", 1.0),
            ("chair", -0.3),
            ("bottle", 0.75),
            ("red cylinder", 1.0),
        ],
        device="cuda",
    )
)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Create figure window
plt.ion()  # Enable interactive mode
fig, ax = plt.subplots()
plt.show()

while True:
    try:
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to frame
            image = Image.fromarray(frame_rgb)
            trav_mask = pipeline(image=image)

            # Clear previous frame
            ax.clear()

            # Display new frame
            ax.imshow(image)
            ax.imshow(trav_mask.cpu(), alpha=0.3, cmap="jet")
            ax.set_xticks([])
            ax.set_yticks([])

            # Update the plot
            plt.pause(0.01)

            # Check if window was closed
            if not plt.get_fignums():
                break

    except KeyboardInterrupt:
        break

# Cleanup
cap.release()
plt.close("all")
