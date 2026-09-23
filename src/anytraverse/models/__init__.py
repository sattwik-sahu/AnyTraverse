"""
Ready-made model wrappers backed by Hugging Face ``transformers``.

Install the ``hf`` extra to use them: ``pip install "anytraverse[hf]"``.
"""

from anytraverse.models._base import HuggingFaceModel
from anytraverse.models.clip import CLIPImageEncoder
from anytraverse.models.clipseg import CLIPSegAttentionMapping
from anytraverse.models.dinov2 import DINOv2ImageEncoder
from anytraverse.models.grounded_sam2 import GroundedSAM2AttentionMapping
from anytraverse.models.owlv2_sam2 import OWLv2SAM2AttentionMapping
from anytraverse.models.sam3 import SAM3AttentionMapping
from anytraverse.models.siglip2 import SigLIP2ImageEncoder

__all__ = [
    "CLIPImageEncoder",
    "CLIPSegAttentionMapping",
    "DINOv2ImageEncoder",
    "GroundedSAM2AttentionMapping",
    "HuggingFaceModel",
    "OWLv2SAM2AttentionMapping",
    "SAM3AttentionMapping",
    "SigLIP2ImageEncoder",
]
