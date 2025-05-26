from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, RANSACRegressor
from typing_extensions import override

PlaneParameters = Tuple[float, float, float, float]


def plane_params_to_euler(a: float, b: float, c: float) -> Tuple[float, float, float]:
    """
    Calculates the yaw, pitch and roll of the camera given the plane
    fitted to the ground.

    It performs a cross product of the plane normal with the y-axis
    to obtain the vector in the direction of the plane, pointing in the
    positive x-direction (world frame).

    This vector is used to calculate the yaw, pitch and roll. Here, the
    implementation uses direct formulae on the plane normal vector components
    to obtain the resultant parameters.

    Args:
        a, b, c (float):
            The parameters for the plane given by `ax + by + cz + d = 0`.
            The vector `[a, b, c]` gives the plane normal.

    Returns:
        Tuple[float, float, float]: The yaw, pitch and roll of the camera.
    """
    # Normalize the plane parameters
    v = np.array([a, b, c])
    v /= np.linalg.norm(v)
    a, b, c = v

    # Calculate pitch (rotation around x-axis)
    roll: float = np.arcsin(-b)

    # Calculate yaw (rotation around y-axis)
    pitch: float = np.arctan2(a, c)

    # Roll is typically zero if the camera's z-axis aligns with the normal
    # We assume no yaw since we align the z-axis with the plane normal
    yaw: float = 0

    return yaw, pitch, roll


class PlaneFitter(ABC):
    _name: str

    def __init__(self, name: str = "plane_fitter", *args, **kwargs) -> None:
        self._name = name

    @abstractmethod
    def fit(self, points: torch.Tensor) -> PlaneParameters:
        pass


class RansacPlaneFitter(PlaneFitter):
    """
    Plane fitter that uses RANSAC to robustly fit a plane to 3D points.

    Attributes:
        _ransac (RANSACRegressor): RANSAC model used for robust plane fitting.
    """

    _ransac: RANSACRegressor

    def __init__(
        self,
        estimator: Any | None = None,
        residual_threshold: float = 0.001,
        max_trials: int = 1_000,
    ) -> None:
        """
        Initializes the RansacPlaneFitter with specified parameters.

        Args:
            estimator (Any | None): Estimator for the RANSAC algorithm (defaults to LinearRegression).
            residual_threshold (float): Maximum residual for a point to be considered an inlier.
            max_trials (int): Maximum number of trials for RANSAC.
        """
        super().__init__(name="plane_fitter__ransac")
        estimator = estimator or LinearRegression()
        self._ransac = RANSACRegressor(
            estimator=estimator,
            residual_threshold=residual_threshold,
            max_trials=max_trials,
        )

    @override
    def fit(self, points: torch.Tensor) -> PlaneParameters:
        """
        Fits a plane to a set of 3D points using the RANSAC algorithm.

        Args:
            points (torch.Tensor): A tensor of shape (N, 3) containing 3D points.

        Returns:
            PlaneParameters: Parameters (a, b, c, d) of the plane ax + by + cz + d = 0.
        """
        # Convert the torch tensor to numpy for compatibility with sklearn
        points_np = points.cpu().numpy()
        X, y = points_np[:, :2], points_np[:, 2]
        self._ransac.fit(X=X, y=y)
        a, b = self._ransac.estimator_.coef_
        c = -1
        d = self._ransac.estimator_.intercept_
        return a, b, c, d


class PCAPlaneFitter(PlaneFitter):
    """
    Plane fitter that uses Principal Component Analysis (PCA) to fit a plane to 3D points.

    This method finds the plane that minimizes the perpendicular distances from the points.
    """

    def __init__(self) -> None:
        """
        Initializes the PCAPlaneFitter.
        """
        super().__init__(name="plane_fitter__pca")

    @override
    def fit(self, points: torch.Tensor) -> PlaneParameters:
        """
        Fits a plane to a set of 3D points using PCA.

        Args:
            points (torch.Tensor): A tensor of shape (N, 3) containing 3D points.

        Returns:
            PlaneParameters: Parameters (a, b, c, d) of the plane ax + by + cz + d = 0.
        """
        # Convert the torch tensor to numpy for compatibility with numpy and scipy
        points_np = points.cpu().numpy()

        # Center the points to calculate PCA (shift to the mean)
        centroid = np.mean(points_np, axis=0)
        centered_points = points_np - centroid

        # Perform PCA to find the normal of the best-fit plane
        pca = PCA(n_components=3)
        pca.fit(centered_points)

        # The normal vector to the plane is the eigenvector corresponding to the smallest eigenvalue
        normal_vector = pca.components_[-1]  # Smallest component

        # Extract a, b, c from the normal vector
        a, b, c = normal_vector

        # Calculate d using the point-normal form of a plane
        d = -np.dot(normal_vector, centroid)

        return a, b, c, d
