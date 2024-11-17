import torch
import numpy as np
from PIL.Image import Image
import time

# create a timeit decorator
# def timeit(method):
#     def timed(*args, **kw):
#         ts = time.time()
#         result = method(*args, **kw)
#         te = time.time()
#         print(f"{method.__name__} took: {te-ts} seconds")
#         return result
#     return timed


class PointcloudManager:
    _fx: float
    _fy: float
    _cx: float
    _cy: float
    _cam_ext: torch.Tensor

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        cam_ext: torch.Tensor | None = None,
    ) -> None:
        """
        Initialize the PointcloudManager with camera intrinsic parameters.

        Args:
            fx (float): Focal length in the x-direction (in pixels).
            fy (float): Focal length in the y-direction (in pixels).
            cx (float): X-coordinate of the principal point (optical center).
            cy (float): Y-coordinate of the principal point (optical center).
        """
        self._fx = fx
        self._fy = fy
        self._cx = cx
        self._cy = cy
        self._cam_ext = cam_ext if cam_ext is not None else torch.eye(4)

    # @timeit
    def depth_to_point_cloud(self, depth_image: torch.Tensor) -> torch.Tensor:
        """
        Convert a depth image tensor to a 3D point cloud using GPU acceleration.

        This function takes a 2D depth image tensor and uses the camera parameters
        to generate a 3D point cloud. The computation is performed entirely with
        PyTorch operations on the GPU.

        Args:
            depth_image (torch.Tensor): A 2D PyTorch tensor representing the depth image.
                Shape should be (height, width).

        Returns:
            torch.Tensor: A 3D point cloud tensor of shape (N, 3), where N is the
                number of points (height * width), and each point is represented
                by its (x, y, z) coordinates.

        Raises:
            ValueError: If the input depth_image is not a 2D PyTorch tensor.
        """
        if depth_image.dim() != 2:
            raise ValueError("Input depth_image must be a 2D PyTorch tensor.")

        device = depth_image.device
        height, width = depth_image.shape

        # Create meshgrid of pixel coordinates
        v, u = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )

        # Reshape depth image
        z = depth_image.reshape(-1)

        # Calculate x and y coordinates
        x = (u.reshape(-1) - self._cx) * z / self._fx
        y = (v.reshape(-1) - self._cy) * z / self._fy

        points = torch.stack((-z, x, -y), dim=1)
        return points

    # @timeit
    def _correct_tilt(self, points: torch.Tensor) -> torch.Tensor:
        """Apply camera extrinsic transformation to correct point cloud orientation.

        This function transforms the point cloud from camera coordinates to world
        coordinates using the camera extrinsic matrix.

        Args:
            points (torch.Tensor): Point cloud in camera coordinates.
                Shape: (N, 3) where N is number of points

        Returns:
            torch.Tensor: Transformed point cloud in world coordinates.
                Shape: (N, 3)
        """
        device = points.device

        # Ensure cam_ext is on the same device as points
        cam_ext = self._cam_ext.to(device)

        # Convert to homogeneous coordinates
        ones = torch.ones(points.shape[0], 1, device=device)
        points_homog = torch.cat([points, ones], dim=1)  # Shape: (N, 4)

        # Apply transformation
        points_world = torch.matmul(points_homog, cam_ext.T)  # Shape: (N, 4)

        # Convert back to 3D coordinates
        points_world = points_world[:, :3] / points_world[:, 3:4]

        return points_world

    # @timeit
    def point_cloud_to_depth_map(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """
        Convert a 3D point cloud to a depth map representation.

        This method takes a 3D point cloud tensor and projects it onto a 2D plane
        to create a depth map.

        Args:
            point_cloud (torch.Tensor): A 3D PyTorch tensor representing the point cloud.
                Shape should be (N, 3), where N is the number of points and each point
                is represented by its (x, y, z) coordinates.

        Returns:
            torch.Tensor: A depth map tensor of shape (N, 3), where each row contains
                (u, v, z) values. 'u' and 'v' are the pixel coordinates in the depth map,
                and 'z' is the corresponding depth value.

        Raises:
            ValueError: If the input point_cloud is not a 2D PyTorch tensor with 3 columns.
        """
        if point_cloud.dim() != 2 or point_cloud.shape[1] != 3:
            raise ValueError(
                "Input point_cloud must be a 2D PyTorch tensor with shape (N, 3)."
            )

        # Extract x, y, z coordinates from the point cloud
        x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]

        # Calculate u and v coordinates (pixel coordinates in the depth map)
        u = (x * self._fx / z) + self._cx
        v = (y * self._fy / z) + self._cy

        # Stack u, v, and z to create the depth map representation
        return torch.stack((u, v, z), dim=1)

    @property
    def cam_extrinsics(self) -> torch.Tensor:
        return self._cam_ext

    @cam_extrinsics.setter
    def cam_extrinsics(self, cam_ext: torch.Tensor):
        assert cam_ext.shape == (4, 4)
        self._cam_ext = cam_ext


# @timeit
def correct_points_with_plane_parms(
    points: torch.Tensor, a: float, b: float, c: float, d: float
) -> torch.Tensor:
    """
    Transform a point cloud so that the plane ax + by + cz + d = 0 becomes the XY plane.

    Parameters:
    points: numpy array of shape (N, 3) containing the point cloud
    a, b, c, d: parameters of the plane equation ax + by + cz + d = 0

    Returns:
    transformed_points: numpy array of shape (N, 3) containing the transformed point cloud
    """
    # Normalize the plane normal vector
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)

    # Find a point on the plane
    # We can find this by solving the equation for any coordinate
    # Let's set x=y=0 and solve for z
    p_on_plane = (
        np.array([0, 0, -d / c])
        if c != 0
        else np.array([-d / a, 0, 0])
        if a != 0
        else np.array([0, -d / b, 0])
    )

    # Calculate the rotation matrix that aligns the normal vector with the z-axis
    # First, find the axis of rotation (cross product of normal and z-axis)
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(normal, z_axis)

    # If normal is parallel to z-axis, no rotation is needed around this axis
    if np.allclose(rotation_axis, 0):
        if normal[2] > 0:
            rotation_matrix = np.eye(3)
        else:
            # If normal points in -z direction, rotate 180 degrees around x-axis
            rotation_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    else:
        # Normalize rotation axis
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

        # Calculate rotation angle
        cos_theta = np.dot(normal, z_axis)
        sin_theta = np.sqrt(1 - cos_theta**2)

        # Build rotation matrix using Rodrigues' rotation formula
        K = np.array(
            [
                [0, -rotation_axis[2], rotation_axis[1]],
                [rotation_axis[2], 0, -rotation_axis[0]],
                [-rotation_axis[1], rotation_axis[0], 0],
            ]
        )
        rotation_matrix = np.eye(3) + sin_theta * K + (1 - cos_theta) * np.dot(K, K)

    # Center the point cloud by subtracting the point on plane
    centered_points = points - p_on_plane

    # Apply rotation
    transformed_points = np.dot(centered_points, rotation_matrix.T)

    return torch.tensor(transformed_points, device=points.device)
