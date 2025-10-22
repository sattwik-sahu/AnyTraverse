import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter


class GridCostmap:
    """
    A 2D costmap that accumulates point-wise cost values from 3D pointclouds,
    visualized from the top (X forward, Y lateral, Z height).
    The grid is laid out so that:
      - X=0 is at the bottom of the grid (rows increase upward).
      - Y=0 is at the center of the grid (cols go from -y_bound to +y_bound).
    """

    def __init__(self, x_bound: float, y_bound: float, resolution: float) -> None:
        """
        Initialize the grid costmap.
        Args:
            x_bound: Maximum X value in meters (min is 0).
            y_bound: Half-width in Y direction (grid covers -y_bound to +y_bound).
            resolution: Size of each grid cell in meters.
        """
        self._resolution = resolution
        self._x_bounds = (0.0, x_bound)
        self._y_bounds = (-y_bound, y_bound)

        self._width = int(np.ceil((self._y_bounds[1] - self._y_bounds[0]) / resolution))
        self._height = int(
            np.ceil((self._x_bounds[1] - self._x_bounds[0]) / resolution)
        )
        self._grid = np.zeros((self._height, self._width), dtype=np.float32)

    def _get_valid(
        self, points: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.float32]]:
        """
        Returns the indexes of the valid points as a binary mask, along with
        a list of the valid points.

        Args:
            points (npt.NDArray[np.float32]): The points to check for. Shape: `(N, 3)`

        Returns:
            tuple[npt.NDArray[np.bool_], npt.NDArray[np.float32]]:
                - The indexes of the valid points as a boolean mask with
                    `True` for valid points and `False` for invalid.
                    Shape: `(N,)`
                - The valid points. Shape: `(M, 3)`

        Note:
            - `N`: Number of points
            - `M`: Number of valid points
        """
        x0, x1 = self._x_bounds
        y0, y1 = self._y_bounds
        xs, ys = points[:, :2].T
        valid_mask: npt.NDArray[np.bool_] = (
            (xs >= x0) & (ys >= y0) & (xs < x1) & (ys < y1)
        )
        valid_points: npt.NDArray[np.float32] = points[valid_mask]
        return valid_mask, valid_points

    def _convert_pointcloud_to_indices(
        self, points: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.int16]:
        """
        Convert 3D points to grid indices (row, col) with bounds filtering.
        Args:
            points: (N, 3) array of [x, y, z].
        Returns:
            (M, 2) array of (row, col) indices.
        """
        if points.shape[0] == 0:
            return np.empty((0, 2), dtype=np.int16)

        shifted = points[:, :2] - np.array(
            [self._x_bounds[0], self._y_bounds[0]], dtype=np.float32
        )
        indices = np.floor_divide(shifted, self._resolution).astype(np.int16)

        rows = self._height - 1 - indices[:, 0]  # X → row (top-down)
        cols = indices[:, 1]  # Y → col (left to right)

        return np.stack([rows, cols], axis=1)

    def grid_indices_to_world(
        self, indices: npt.NDArray[np.int32]
    ) -> npt.NDArray[np.float32]:
        """
        Convert an array of grid (row, col) indices to world (x, y) coordinates.
        Args:
            indices: (K, 2) array of (row, col) indices.
        Returns:
            (K, 2) array of (x, y) coordinates in meters.
        """
        rows, cols = indices[:, 0], indices[:, 1]

        x = self._x_bounds[0] + (self._height - 1 - rows + 0.5) * self._resolution
        y = self._y_bounds[0] + (cols + 0.5) * self._resolution

        return np.stack([x, y], axis=1).astype(np.float32)

    def update_costmap(
        self,
        points: npt.NDArray[np.float32],
        costs: npt.NDArray[np.float32],
        sigma: float = 1.0,
        alpha: float = 0.75,
    ) -> npt.NDArray[np.int16]:
        """
        Update the grid costmap with point costs and apply Gaussian smoothing.
        Args:
            points: (N, 3) array of [x, y, z] pointcloud.
            costs: (N,) array of scalar values (e.g., traversability scores), one per point.
            sigma: Standard deviation for Gaussian blur in grid cells.
        """
        assert points.shape[0] == costs.shape[0], (
            "Number of costs must match number of points."
        )

        valid_mask, valid_points = self._get_valid(points=points)
        points = valid_points
        costs = costs[valid_mask]

        valid_inxs = self._convert_pointcloud_to_indices(points).T
        rows, cols = valid_inxs

        temp_grid = np.zeros_like(self._grid, dtype=np.float32)
        temp_grid[rows, cols] = costs

        self._grid = self._grid * alpha + (1 - alpha) * gaussian_filter(
            temp_grid, sigma=sigma
        )
        return valid_inxs

    def grid_index_to_point2d(self, row: int, col: int) -> tuple[float, float]:
        """
        Convert a grid index back to the corresponding (x, y) coordinate in meters.
        Args:
            row: Row index in the grid (0 at top).
            col: Column index in the grid (0 at leftmost Y = -y_bound).
        Returns:
            (x, y): Real-world coordinates (meters), at the center of the cell.
        """
        x = self._x_bounds[0] + (self._height - 1 - row + 0.5) * self._resolution
        y = self._y_bounds[0] + (col + 0.5) * self._resolution
        return (x, y)

    def get_grid(self) -> npt.NDArray[np.float32]:
        """
        Return the current grid values.
        Returns:
            2D costmap as a NumPy array.
        """
        return self._grid.copy()
