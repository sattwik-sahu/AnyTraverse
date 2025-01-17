from transformers import (
    pipeline,
    Pipeline,
    AutoImageProcessor,
    AutoModelForDepthEstimation,
)
from dataclasses import dataclass
from PIL.Image import Image
from typing import List, Literal, Tuple
import torch
# import time

# define timeit decorator
# def timeit(method):
#     def timed(*args, **kw):
#         ts = time.time()
#         result = method(*args, **kw)
#         te = time.time()
#         print(f"{method.__name__} took: {te-ts} seconds")
#         return result
#     return timed


@dataclass
class DepthAnythingOutput:
    tensor: torch.Tensor
    image: Image | List[Image]


class DepthAnythingV2_small:
    _pipeline: Pipeline

    def __init__(self, device: Literal["cpu", "cuda"]) -> None:
        self._pipeline = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
            device=device,
        )

    def _postprocess(
        self, y: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        y = y.unsqueeze(1)
        y_resized = torch.nn.functional.interpolate(
            y,
            size=image_size,
            mode="bicubic",
            align_corners=True,
        )
        return y_resized

    def __call__(self, x: Image | List[Image]) -> DepthAnythingOutput:
        """
        Runs the depth-anything-v2-small model on the given image or list of
        images and returns the tensor(s) and depth image(s).

        Args:
            x (Image | List[Image]): The image or list of images.

        Returns:
            DepthAnythingOutput: The output which is just a dataclass
                wrapper around the actual output from the HuggingFace model.
        """
        outputs = self._pipeline(x)
        return DepthAnythingOutput(
            tensor=self._postprocess(
                y=outputs["predicted_depth"],  # type: ignore
                image_size=x.size[::-1],  # type: ignore
            ),
            image=outputs["depth"],  # type: ignore
        ), outputs


class DepthAnythingV2_MetricIndoorLarge:
    _pipeline: Pipeline

    def __init__(self, device: Literal["cpu", "cuda"]) -> None:
        self._pipeline = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
            device=device,
        )

    def _postprocess(
        self, y: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        y = y.unsqueeze(1)
        y_resized = torch.nn.functional.interpolate(
            y,
            size=image_size,
            mode="bicubic",
            align_corners=True,
        )
        return y_resized

    def __call__(self, x: Image | List[Image]) -> torch.Tensor:
        """
        Runs the depth-anything-v2-small model on the given image or list of
        images and returns the tensor(s) and depth image(s).

        Args:
            x (Image | List[Image]): The image or list of images of height H and width W.

        Returns:
            torch.Tensor: The ouptut from the model. Dimension: (H, W)
        """
        outputs = self._pipeline(x)
        return outputs.get("depth")  # type: ignore


class DepthAnythingV2_MetricOutdoorLarge:
    _device: Literal["cpu", "cuda", "mps"]

    # @timeit
    def __init__(self, device: Literal["cpu", "cuda", "mps"] | torch.device = "cpu") -> None:
        self._device = device

        self._processor = AutoImageProcessor.from_pretrained(
            "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"
        )
        self._model = AutoModelForDepthEstimation.from_pretrained(
            "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"
        ).to(device=self._device)

    # @timeit
    def __call__(self, x: Image) -> torch.Tensor:
        x_tensor = self._processor(
            images=x, return_tensors="pt").to(self._device)
        type(x_tensor)
        with torch.no_grad():
            outputs = self._model(**x_tensor)
            predicted_depth = outputs.predicted_depth

        # interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=x.size[::-1],
            mode="bicubic",
            align_corners=False,
        )

        return prediction
