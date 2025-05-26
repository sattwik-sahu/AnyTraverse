from pathlib import Path
from anytraverse.utils.datasets.posix_path_dataset import PosixPathDataset
from anytraverse.utils.datasets.processors import RGBTraversibleMaskProcessor
from anytraverse.config.datasets.deepscene import config as deepscene_config


class DeepsceneDataset(PosixPathDataset):
    """
    Deepscene dataset.
    """

    def __init__(
        self,
        images_root: Path | str,
        masks_root: Path | str,
        image_pattern: str,
        mask_pattern: str,
    ) -> None:
        super().__init__(
            images_root=images_root,
            masks_root=masks_root,
            image_pattern=image_pattern,
            mask_pattern=mask_pattern,
            traversible_mask_processor=RGBTraversibleMaskProcessor(
                trav_rgb=deepscene_config.traversibles, non_trav_rgb=[]
            ),
            target_size=(720, 480),
        )
