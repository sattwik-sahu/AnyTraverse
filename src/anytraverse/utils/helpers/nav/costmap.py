import numpy as np
from numpy import typing as npt


class Costmap2D:
    def __init__(self, grid_size: int, cell_size: float, beta: float = 0.9) -> None:
        self._cell_size = cell_size
        self._grid_size = grid_size
        self._costmap = np.zeros((grid_size, grid_size), dtype=np.float32)
        self._origin_index = (self._grid_size // 2, self._grid_size // 2)
        self._beta = beta

    @property
    def costmap(self) -> npt.NDArray[np.float32]:
        return self._costmap

    @property
    def bound(self) -> float:
        return self._cell_size * self._grid_size * 0.5

    def _get_valid_pointcloud_mask(
        self, pointcloud: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.bool_]:
        return (
            (pointcloud[:, 0] > -self.bound)
            & (pointcloud[:, 0] < self.bound)
            & (pointcloud[:, 1] > -self.bound)
            & (pointcloud[:, 1] < self.bound)
        )

    def to_grid_indexes(
        self, pointcloud: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.int32]:
        """
        Convert world coordinates to integer grid indexes.

        Args:
            pointcloud (npt.NDArray[np.float32]): Nx2 array of (x, y) points.

        Returns:
            npt.NDArray[np.int32]: Nx2 array of (row, col) grid indexes.
        """
        origin_x, origin_y = self._origin_index

        # shift by +0.5 so that (0,0) world goes to the center of the origin cell
        grid_inx_x = (
            np.floor_divide(pointcloud[:, 0], self._cell_size).astype(np.int32)
            + origin_x
        )
        grid_inx_y = (
            np.floor_divide(pointcloud[:, 1], self._cell_size).astype(np.int32)
            + origin_y
        )

        return np.stack((grid_inx_x, grid_inx_y), axis=1)

    def to_world_coords(
        self, grid_indexes: npt.NDArray[np.int_]
    ) -> npt.NDArray[np.float32]:
        """
        Converts costmap grid indexes to world coordinates (cell centers).

        Args:
            grid_indexes (npt.NDArray[np.int_]): Nx2 array of (row, col) grid indexes.

        Retrrns:
            npt.NDArray[np.float32]: Nx2 array of (x, y) world coordinates.
        """
        return (grid_indexes - self._origin_index) * self._cell_size

    def update(
        self,
        pointcloud: npt.NDArray[np.float32],
        traversability_scores: npt.NDArray[np.float32],
    ) -> None:
        valid_mask = self._get_valid_pointcloud_mask(pointcloud=pointcloud)
        pointcloud, traversability_scores = (
            pointcloud[valid_mask],
            traversability_scores[valid_mask],
        )

        if pointcloud.shape[0] == 0:
            return

        grid_indexes_to_update = self.to_grid_indexes(pointcloud=pointcloud)
        flat_grid_indexes = (
            grid_indexes_to_update[:, 0] * self._grid_size
            + grid_indexes_to_update[:, 1]
        )

        traversability_sums = np.bincount(
            flat_grid_indexes,
            weights=traversability_scores,
            minlength=self._grid_size**2,
        )
        n_points_in_grid = np.bincount(flat_grid_indexes, minlength=self._grid_size**2)
        mean_traversability_scores = np.zeros_like(traversability_sums)
        flat_grid_indexes_with_points = n_points_in_grid > 0
        mean_traversability_scores[flat_grid_indexes_with_points] = (
            traversability_sums[flat_grid_indexes_with_points]
            / n_points_in_grid[flat_grid_indexes_with_points]
        )
        flat_costmap = self._costmap.ravel()
        flat_costmap[flat_grid_indexes_with_points] = (1 - self._beta) * flat_costmap[
            flat_grid_indexes_with_points
        ] + self._beta * mean_traversability_scores[flat_grid_indexes_with_points]
