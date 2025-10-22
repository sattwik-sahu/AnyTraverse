import torch
import numpy as np


def euler_to_rotation_matrix(
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    degrees: bool = True,
) -> torch.Tensor:
    """Converts euler angles to a 4x4 rotation matrix.

    Creates a rotation matrix from yaw (rotation around Z axis), pitch (rotation around Y axis),
    and roll (rotation around X axis) angles. The rotations are applied in the order: yaw,
    pitch, roll. If any angle is not provided, it is assumed to be zero.

    Params:
        yaw (float, optional): Rotation angle around the Z axis. Defaults to None (0)
        pitch (float, optional): Rotation angle around the Y axis. Defaults to None (0)
        roll (float, optional): Rotation angle around the X axis. Defaults to None (0)
        degrees (bool, optional): If True, input angles are in degrees, otherwise in radians.
            Defaults to True

    Returns:
        torch.Tensor: A 4x4 rotation matrix as a torch tensor. The last row and column
            are set to [0, 0, 0, 1] to make it a valid transformation matrix

    Example:
        >>> # Create rotation matrix for 90 degree yaw (around Z)
        >>> R = euler_to_rotation_matrix(yaw=90)
        >>>
        >>> # Create matrix for compound rotation (45° yaw, 30° pitch, 15° roll)
        >>> R = euler_to_rotation_matrix(yaw=45, pitch=30, roll=15)
        >>>
        >>> # Using radians
        >>> R = euler_to_rotation_matrix(yaw=np.pi/2, degrees=False)
    """
    # Convert angles to radians if necessary
    if degrees:
        yaw = np.deg2rad(yaw)
        pitch = np.deg2rad(pitch)
        roll = np.deg2rad(roll)

    # Compute trigonometric values
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cr = np.cos(roll)
    sr = np.sin(roll)

    # Create rotation matrices for each axis
    # Rotation around Z axis (yaw)
    Rz = torch.tensor([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=torch.float32)

    # Rotation around Y axis (pitch)
    Ry = torch.tensor([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=torch.float32)

    # Rotation around X axis (roll)
    Rx = torch.tensor([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=torch.float32)

    # Combine rotations: R = Rz @ Ry @ Rx
    # R = Rz @ Ry @ Rx
    R = Rx @ Ry @ Rz

    # Create 4x4 transformation matrix
    transform = torch.eye(4)
    transform[:3, :3] = R

    return transform


def correct_rotation_rpy(
    points: torch.Tensor,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    degrees: bool = True,
) -> torch.Tensor:
    """
    Corrects 3D points using the roll, pitch and yaw of the camera frame wrt global frame.

    Args:
        points (torch.Tensor): The set of points to correct. Shape: (N, 3)
        yaw (float): The yaw of the camera frame.
        pitch (float): The pitch of the camera frame.
        roll (float): The roll of the camera frame.
        degrees (bool): Whether the values of roll, pitch and yaw provided
            are in degrees. If false, uses radians. Default: True

    Returns:
        corrected_points (torch.Tensor): The set of corrected points,
            in the world frame. Shape: (N, 3)
    """
    device = points.device
    cam_ext = euler_to_rotation_matrix(yaw=yaw, pitch=pitch, roll=roll, degrees=degrees)

    # Ensure cam_ext is on the same device as points
    cam_ext = cam_ext.to(device)

    # Convert to homogeneous coordinates
    ones = torch.ones(points.shape[0], 1, device=device)
    points_homog = torch.cat([points, ones], dim=1)  # Shape: (N, 4)

    # Apply transformation
    points_world = torch.matmul(points_homog, cam_ext.T)  # Shape: (N, 4)

    # Convert back to 3D coordinates
    points_world = points_world[:, :3] / points_world[:, 3:4]

    return points_world
