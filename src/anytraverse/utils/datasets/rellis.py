import numpy as np
from numpy import typing as npt
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, Literal
from PIL import Image
from anytraverse.utils.datasets.processors import ValueTraversibleMaskProcessor
from anytraverse.config.datasets.rellis import config as rellis_config


class RellisDataset(Dataset):
    """
    Rellis-3D dataset.

    Attributes:
        _images_root (Path): Path to the images directory.
        _masks_root (Path): Path to the masks directory.
        _paths (str): List of paths to the images and masks.
    """

    _images_root: Path
    _masks_root: Path
    _paths: List[Tuple[str, str]]
    _traversible_mask_processor: ValueTraversibleMaskProcessor

    def __init__(
        self,
        images_root: Path | str,
        masks_root: Path | str,
        path_list_path: Path | str,
    ) -> None:
        self.images_root = Path(images_root)
        self.masks_root = Path(masks_root)
        self._paths = []
        # self.resolution = (400, 640)
        with open(path_list_path, "r") as f:
            for line in f:
                image_path, mask_path = line.strip().split(" ")
                self._paths.append((image_path, mask_path))

        self._traversible_mask_processor = ValueTraversibleMaskProcessor(
            trav_values=rellis_config.traversibles, non_trav_values=[]
        )

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(
        self, idx: int
    ) -> dict[Literal["image", "mask"], npt.NDArray[np.int16] | npt.NDArray[np.bool_]]:
        image_path = self.images_root.joinpath(self._paths[idx][0])
        mask_path = self.masks_root.joinpath(self._paths[idx][1])

        image = Image.open(image_path)
        mask = Image.open(mask_path)

        mask_traversible = self._traversible_mask_processor(mask)

        return {"image": np.array(image), "mask": mask_traversible}
