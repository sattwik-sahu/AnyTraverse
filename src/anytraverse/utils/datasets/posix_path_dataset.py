import numpy as np
from numpy import typing as npt
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Literal
from PIL import Image
from anytraverse.utils.datasets import ImageProcessor


class PosixPathDataset(Dataset):
    """
    A posix path based dataset which takes the images and masks root
    posix paths patterns and creates a dataset by extracting the images and
    corresponding masks.

    It processes the masks using the provided mask processor.

    Attributes:
        _images_path_pattern (str): Path to the images directory.
        _masks_root (Path): Path to the masks directory.
        _target_size (Tuple[int, int]): The target size of the images. Value
            must be given as a tuple of (width, height).
    """

    _images_root: Path
    _masks_root: Path
    _traversible_mask_processor: ImageProcessor
    _target_size: Tuple[int, int]

    def __init__(
        self,
        images_root: Path | str,
        masks_root: Path | str,
        image_pattern: str,
        mask_pattern: str,
        traversible_mask_processor: ImageProcessor,
        target_size: Tuple[int, int] | None = None,
    ) -> None:
        self.images_root, self._image_pattern = Path(images_root), image_pattern
        self.masks_root, self._mask_pattern = Path(masks_root), mask_pattern
        self._traversible_mask_processor = traversible_mask_processor
        self._paths = []

        self._paths = [
            (image_path.absolute(), mask_path.absolute())
            for image_path, mask_path in zip(
                sorted(self.images_root.glob(self._image_pattern)),
                sorted(self.masks_root.glob(self._mask_pattern)),
            )
        ]

        if target_size is not None:
            self._target_size = target_size
        else:
            # Set the target size to the size of the first image
            self._target_size = Image.open(self._paths[0][0]).size

    def __len__(self) -> int:
        return len(self._paths)

    def _resize_image_and_mask(
        self, image: Image.Image, mask: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Resizes two PIL.Image objects to the dimensions of the smaller image.

        Args:
            image (Image.Image): The image.
            mask (Image.Image): The mask.

        Returns:
            Tuple[Image.Image, Image.Image]: A tuple of the resized images.
        """
        # Resize both images to the smallest dimensions
        image1_resized = image.resize(
            size=self._target_size, resample=Image.Resampling.LANCZOS
        )
        image2_resized = mask.resize(
            size=self._target_size, resample=Image.Resampling.LANCZOS
        )

        return image1_resized, image2_resized

    def __getitem__(
        self, idx: int
    ) -> dict[Literal["image", "mask"], npt.NDArray[np.int16] | np.ndarray]:
        image_path = self._paths[idx][0]
        mask_path = self._paths[idx][1]

        image = Image.open(image_path)
        mask = Image.open(mask_path)
        if image.size != self._target_size:
            image, mask = self._resize_image_and_mask(image, mask)
        mask_traversible = self._traversible_mask_processor(mask)

        return {"image": np.array(image), "mask": mask_traversible}
