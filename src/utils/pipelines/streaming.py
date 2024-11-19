import cv2
from fastapi.responses import HTMLResponse
import numpy as np
import asyncio
import base64
import json
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

# import EasyPySpin
from PIL import Image
import torch
from typing import Optional, Tuple
from dataclasses import dataclass
from fastapi.staticfiles import StaticFiles
import os
from contextlib import asynccontextmanager

from config.utils import CameraConfig, HeightScoringConfig, PlaneFittingConfig
from utils.helpers.plane_fit import PCAPlaneFitter
from utils.models.clipseg.pooler import WeightedMaxPooler, ProbabilisticPooler
from utils.pipelines.pipeline_002 import Pipeline2
from config.pipeline_002 import PipelineConfig

# Force OpenCV to use CPU
os.environ["OPENCV_DNN_BACKEND_CUDA"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


@dataclass
class StreamConfig:
    overlay_color: Tuple[int, int, int] = (0, 255, 0)  # BGR format
    alpha: float = 0.3
    frame_width: int = 640
    frame_height: int = 480


def create_pipeline():
    fx, fy, cx, cy = 2813.643275, 2808.326079, 969.285772, 624.049972
    device = "cuda" if torch.cuda.is_available() else "mps"

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
    )

    return Pipeline2(config=config)


class CameraStream:
    def __init__(self, config: StreamConfig):
        self.config = config
        self.pipeline = create_pipeline()
        # self.cap = EasyPySpin.VideoCapture(0)
        self.cap = cv2.VideoCapture(0)

        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
        self.running = False

    def overlay_mask(
        self, frame: np.ndarray, mask: np.ndarray, threshold: float = 0.5
    ) -> np.ndarray:
        mask_binary = mask > threshold
        color_mask = np.zeros_like(frame)
        color_mask[mask_binary] = self.config.overlay_color
        return cv2.addWeighted(frame, 1, color_mask, self.config.alpha, 0)

    async def get_frame(self) -> Optional[str]:
        ret, frame = self.cap.read()
        if not ret:
            return None

        frame = frame.astype(np.uint8)  # type: ignore

        frame = cv2.resize(
            src=frame,
            dsize=(self.config.frame_width, self.config.frame_height),
            fx=self.pipeline._config.camera.fx,
            fy=self.pipeline._config.camera.fy,
        )
        image = Image.fromarray(frame)

        trav_mask = self.pipeline(image=image)
        mask_np = trav_mask.cpu().numpy()

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        overlay = self.overlay_mask(frame, mask_np)

        _, buffer = cv2.imencode(".jpg", overlay)
        return base64.b64encode(buffer).decode("utf-8")

    def release(self):
        self.cap.release()


# Global variables
stream_config = StreamConfig()
camera_stream = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global camera_stream
    camera_stream = CameraStream(stream_config)
    yield
    # Shutdown
    if camera_stream:
        camera_stream.release()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files after the WebSocket route
# app.mount("/static", StaticFiles(directory="data/ui/static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("data/ui/index.html", "r") as f:
        return f.read()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    global camera_stream

    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                config = json.loads(data)
                stream_config.overlay_color = tuple(
                    config.get("color", stream_config.overlay_color)
                )
                stream_config.alpha = config.get("alpha", stream_config.alpha)
            except asyncio.TimeoutError:
                pass

            if camera_stream:
                frame = await camera_stream.get_frame()
                if frame:
                    await websocket.send_text(frame)
                await asyncio.sleep(0.03)
    except Exception as e:
        print(f"WebSocket error: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
