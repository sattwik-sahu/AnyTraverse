from pathlib import Path

from anytraverse.config.datasets.rugd import config as rugd_config
from anytraverse.utils.datasets.posix_path_dataset import PosixPathDataset
from anytraverse.utils.datasets.processors import RGBTraversibleMaskProcessor


class RUGD_Dataset(PosixPathDataset):
    """
    RUGD dataset.
    """

    def __init__(
        self,
        images_root: Path | str,
        masks_root: Path | str,
        image_pattern: str,
        mask_pattern: str,
        **kwargs,
    ) -> None:
        super().__init__(
            images_root=images_root,
            masks_root=masks_root,
            image_pattern=image_pattern,
            mask_pattern=mask_pattern,
            traversible_mask_processor=RGBTraversibleMaskProcessor(
                trav_rgb=rugd_config.traversibles, non_trav_rgb=[]
            ),
            **kwargs,
        )
