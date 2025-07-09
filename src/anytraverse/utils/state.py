from dataclasses import dataclass
from enum import Enum
from anytraverse import typing as anyt


class TraversalState(Enum):
    UNKOWN_OBJ = 1
    UNKNOWN_SCENE = 2
    OK = 3


@dataclass
class Threshold:
    ref_scene_similarity: float
    roi_uncertainty: float


@dataclass
class AnyTraverseState:
    """
    Contains the state of the AnyTraverse pipeline on one timestep,
    operating on an image.
    """

    image_encoding: anyt.Encoding
    """The image encoding of the image passed"""

    attention_maps: list[anyt.PromptAttentionMap]
    """
    The attention maps on the image for each of the prompts,
    each a `anyt.AttentionMap` (alias for `torch.Tensor`)
    of shape `(H, W)`, where the image passed in was had dimensions
    `W x H`
    """

    traversability_map: anyt.TraversabilityMap
    """
    The **traversability map**, a `anyt.TraversabilityMap` (alias
    for `torch.Tensor`) of shape `(H, W)` where each pixel gives the
    traversability score of that pixel between `0` (most untraversable)
    to `1` (most traversable)
    """

    uncertainty_map: anyt.UncertaintyMap
    """
    The uncertainty map, a `anyt.UncertaintyMap` (alias for `torch.Tensor`)
    of shape `(H, W)` where each pixel gives the uncertainty score of that
    pixel between `0` (most certain) to `1` (most uncertain)
    """

    traversability_map_roi: anyt.TraversabilityMap
    """
    The region of interest (ROI) extracted from the traversability map
    """

    uncertainty_map_roi: anyt.UncertaintyMap
    """
    The region of interest (ROI) extracted from the uncertainty map
    """

    ref_scene_similarity: float
    """
    Similarity score of current scene with the reference scene.
    """

    traversability_preferences: anyt.TraversabilityPreferences
    """
    The traversability preferences as `anyt.TraversabilityPreferences`
    (alias for `dict[str, float]`) of the form `{"<prompt1>": <weight1>, ...}`
    """

    roi_bbox: tuple[tuple[int, int], tuple[int, int]]
    """
    The region of interest (ROI), as start and end coordinates of a bounding box.
    Format: ((x_start, y_start), (x_end, y_end))
    """

    roi_uncertainty: float
    """The mean uncertainty in the region of interest in the image"""

    roi_traversability: float
    """The mean traversability in the region of interest in the image"""

    traversal_state: TraversalState
    """
    The state as a `TraversalState` `enum`, which contains whether it is
    ok (`OK`) to keep going, an unknown scene is encountered (`UNKNOWN_SCENE`),
    or an unknown object has entered the region of interest (`UNKNOWN_OBJ`)
    """
