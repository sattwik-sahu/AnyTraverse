import torch
from anytraverse.utils.pipelines.base import AnyTraverse
from anytraverse.utils.models.clipseg import CLIPSegAttentionMapping
from anytraverse.utils.models.clip import CLIPImageEncoder
from anytraverse.utils.modules.trav_pooling import WeightedMaxTraversabilityPooler
from anytraverse.utils.modules.unc_pooling import (
    InverseMaxProbabilityUncertaintyPooler,
)
from anytraverse.utils.roi import RegionOfInterest
from anytraverse.utils.state import Threshold
from anytraverse import typing as anyt
from PIL import Image as PILImage


def build_pipeline_from_paper(
    init_traversabilty_preferences: anyt.TraversabilityPreferences,
    ref_scene_similarity_threshold: float,
    roi_uncertainty_threshold: float,
    roi_x_bounds: tuple[float, float],
    roi_y_bounds: tuple[float, float],
) -> AnyTraverse[PILImage.Image]:
    """
    Builds the original pipeline from the paper.
    """
    return AnyTraverse[PILImage.Image](
        prompt_attention_mapping=CLIPSegAttentionMapping[PILImage.Image](),
        image_encoder=CLIPImageEncoder[PILImage.Image](
            model_name="openai/clip-vit-base-patch32"
        ),
        traversability_pooler=WeightedMaxTraversabilityPooler,
        uncertainty_pooler=InverseMaxProbabilityUncertaintyPooler,
        roi=RegionOfInterest(x_bounds=roi_x_bounds, y_bounds=roi_y_bounds),
        threshold=Threshold(
            ref_scene_similarity=ref_scene_similarity_threshold,
            roi_uncertainty=roi_uncertainty_threshold,
        ),
        init_traversability_preferences=init_traversabilty_preferences,
        similarity_func=torch.cosine_similarity,
    )
