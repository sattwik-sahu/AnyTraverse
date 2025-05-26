import torch
from PIL import Image
from anytraverse.utils.models.depth_anything.model import (
    DepthAnythingV2_MetricOutdoorLarge,
)
from typing import Literal, Tuple, NamedTuple
from anytraverse.config.pipeline_001 import CameraConfig
from anytraverse.utils.helpers.pointcloud import (
    PointcloudManager,
    correct_points_with_plane_parms,
)
from anytraverse.utils.helpers.plane_fit import PlaneFitter, PlaneParameters
# import time

Device = Literal["cpu", "cuda", "mps"] | torch.device


# define a timeit decorator
# def timeit(method):
#     def timed(*args, **kw):
#         ts = time.time()
#         result = method(*args, **kw)
#         te = time.time()
#         print(f"{method.__name__} took: {te-ts} seconds")
#         return result

#     return timed


class HeightScoringOutput(NamedTuple):
    depth_image: torch.Tensor
    points: torch.Tensor
    plane_params: PlaneParameters
    points_corrected: torch.Tensor
    scores: torch.Tensor


def p_trav(z: torch.Tensor, alpha: float, z_thresh: float) -> torch.Tensor:
    return torch.sigmoid(-alpha * (z - z_thresh))


class HeightScoringPipeline:
    _model: DepthAnythingV2_MetricOutdoorLarge
    _device: Device
    _z_thresh: Tuple[float, float]
    _alpha: Tuple[float, float]
    _pointcloud_manager: PointcloudManager
    _plane_fitter: PlaneFitter

    def __init__(
        self,
        plane_fitter: PlaneFitter,
        z_thresh: Tuple[float, float],
        alpha: Tuple[float, float],
        camera_config: CameraConfig,
        device: Device = "cpu",
    ):
        self._model = DepthAnythingV2_MetricOutdoorLarge(device=device)
        self._device = device
        self._plane_fitter = plane_fitter
        self._z_thresh = z_thresh
        self._alpha = alpha
        self._pointcloud_manager = PointcloudManager(
            fx=camera_config.fx,
            fy=camera_config.fy,
            cx=camera_config.cx,
            cy=camera_config.cy,
        )
        print("HeightScoringPipeline initialized using device:", self._model._device)

    # @timeit
    def _create_pointcloud(self, depth_z: torch.Tensor) -> torch.Tensor:
        points: torch.Tensor = self._pointcloud_manager.depth_to_point_cloud(
            depth_image=depth_z
        )  # Dimensions: (H * W, 3)
        return points

    # @timeit
    def _fit_plane(self, points: torch.Tensor) -> PlaneParameters:
        plane_params = self._plane_fitter.fit(points=points)
        return plane_params

    # @timeit
    def __call__(
        self, image: Image.Image, plane_fit_mask: torch.Tensor
    ) -> HeightScoringOutput:
        """
        Runs the height scoring pipeline on the image and returns the height
        score for each pixel in the image.

        Args:
            image (Image.Image): The image.
            plane_fit_mask (torch.Tensor): The mask of which points to use
                for fitting the plane. Dimensions: (H, W)

        Returns:
            torch.Tensor:
            The height score for each pixel in the image. Dimensions: (H, W)
        """
        # Run depth anything
        depth_z: torch.Tensor = self._model(x=image).squeeze(0, 1)  # Dimensions: (H, W)

        # Create pointcloud
        points: torch.Tensor = self._create_pointcloud(
            depth_z=depth_z
        )  # Dimensions: (H * W, 3)

        # Fit a plane to the points
        a, b, c, d = self._fit_plane(
            points=points[plane_fit_mask.flatten().cpu().numpy()]
        )

        # Correct the points with the plane parameters
        points_corrected: torch.Tensor = correct_points_with_plane_parms(
            points=points.cpu(), a=a, b=b, c=c, d=d
        )  # Dimensions: (H * W, 3)

        # Get the z-coordinates for each pixel
        zs: torch.Tensor = points_corrected[:, 2].reshape(
            image.size[::-1]
        )  # Dimensions: (H, W)

        # Calculate the height scores
        cond_z = zs >= 0
        p_pos = p_trav(z=zs, alpha=self._alpha[1], z_thresh=self._z_thresh[1])
        p_neg = p_trav(z=zs, alpha=-self._alpha[0], z_thresh=self._z_thresh[0])
        height_scores = torch.where(cond_z, p_pos, p_neg)

        return HeightScoringOutput(
            depth_image=depth_z,
            points=points,
            plane_params=(a, b, c, d),
            points_corrected=points_corrected,
            scores=height_scores,
        )
