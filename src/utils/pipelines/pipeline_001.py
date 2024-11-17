from typing import Tuple

import numpy as np
import torch
from PIL.Image import Image
from typing_extensions import override

from config.pipeline_001 import PipelineConfig
from utils.helpers.plane_fit import (
    PlaneParameters,
    plane_params_to_euler,
)
from utils.helpers.pointcloud import PointcloudManager
from utils.helpers.rotation import correct_rotation_rpy
from utils.models.clipseg.model import CLIPSeg
from utils.models.depth_anything.model import (
    DepthAnythingOutput,
    DepthAnythingV2_small,
)
from utils.pipelines import CLIPSegOffnavPipeline


class Pipeline1(CLIPSegOffnavPipeline):
    _clipseg: CLIPSeg
    _depth_anything: DepthAnythingV2_small
    _config: PipelineConfig
    _pointcloud_manager: PointcloudManager

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(name="Pipeline_001")
        self._config = config
        self._clipseg = CLIPSeg(device=self._config.device)
        self._depth_anything = DepthAnythingV2_small(device=self._config.device)
        self._pointcloud_manager = PointcloudManager(
            fx=self._config.camera.fx,
            fy=self._config.camera.fy,
            cx=self._config.camera.cx,
            cy=self._config.camera.cy,
        )

    def _get_pooled_traversibility_mask(self, image: Image) -> torch.Tensor:
        # Run CLIPSeg on the image
        trav_masks = self._clipseg(image=image, prompts=self._config.prompts)
        pooled_mask = trav_masks.max(dim=0).values
        return pooled_mask

    def _get_depth_image(self, image: Image) -> torch.Tensor:
        depth_output: DepthAnythingOutput = self._depth_anything(x=image)
        depth_z: torch.Tensor = torch.exp(0.05 * -depth_output.tensor.squeeze(0, 1))
        return depth_z

    def _get_plane_fit_parameters(
        self, points: torch.Tensor, pooled_trav_mask: torch.Tensor
    ) -> PlaneParameters:
        fitter = self._config.plane_fitting.fitter
        if fitter is not None:
            a, b, c, d = fitter.fit(
                points=points[
                    (
                        pooled_trav_mask.reshape((-1, 1)).cpu().numpy()
                        > self._config.plane_fitting.trav_thresh
                    ).flatten()
                ]
            )
            return a, b, c, d
        else:
            # X-Y Plane: 0x + 0y + z + 0 = 0 => z = 0
            return 0, 0, 1, 0

    def _get_euler_angles(
        self, a: float, b: float, c: float
    ) -> Tuple[float, float, float]:
        yaw, pitch, roll = plane_params_to_euler(a=a, b=b, c=c)
        pitch = -pitch - np.pi
        return yaw, pitch, roll

    def _correct_points(
        self, points: torch.Tensor, pooled_trav_mask: torch.Tensor
    ) -> torch.Tensor:
        a, b, c, d = self._get_plane_fit_parameters(
            points=points, pooled_trav_mask=pooled_trav_mask
        )
        yaw, pitch, roll = self._get_euler_angles(a=a, b=b, c=c)
        points_corr = correct_rotation_rpy(
            points=points - torch.tensor([[0, 0, d]]),
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            degrees=False,
        )
        points_corr[:, 0] *= -1
        return points_corr

    def _get_height_scores(
        self, point_zs: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        zs = point_zs.reshape(shape=image_size).to(device=self._config.device)
        z_thresh = self._config.scoring.height.z_thresh
        height_scores = torch.zeros_like(zs)
        height_scores[zs <= z_thresh * 1e-3] = 1.0
        return height_scores

    def _combine_traversibility_scores(
        self, pooled_trav_mask: torch.Tensor, height_scores: torch.Tensor
    ) -> torch.Tensor:
        scores = torch.exp(
            self._config.scoring.beta * torch.log(pooled_trav_mask[0])
            + (1 - self._config.scoring.beta) * torch.log(height_scores)
        )
        return scores

    @override
    def __call__(self, image: Image) -> torch.Tensor:
        pooled_trav_mask: torch.Tensor = self._get_pooled_traversibility_mask(
            image=image
        )
        depth_image: torch.Tensor = self._get_depth_image(image=image)
        points = self._pointcloud_manager.depth_to_point_cloud(depth_image=depth_image)
        points_corr = self._correct_points(
            points=points, pooled_trav_mask=pooled_trav_mask
        )
        image_width, image_height = image.size
        height_scores = self._get_height_scores(
            point_zs=points_corr[:, 2], image_size=(image_height, image_width)
        )
        trav_scores = self._combine_traversibility_scores(
            pooled_trav_mask=pooled_trav_mask, height_scores=height_scores
        )

        return trav_scores
