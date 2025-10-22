from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation, PreTrainedModel
from typing import Tuple, Dict, Any


def load_clipseg_processor_and_model(
    pretrained_model_name: str | None = None,
) -> Tuple[CLIPSegProcessor | Tuple[CLIPSegProcessor, Dict[str, Any]], PreTrainedModel]:
    """
    Loads and returns the processor and pretrained model for CLIPSeg.

    Args:
        pretrained_model_name (str | None): The name of the pretrained model
            to load for CLIPSeg. Defaults to None in which case the model
            `CIDAS/clipseg-rd64-refined` is used.

    Returns:
        Tuple[CLIPSegProcessor | Tuple[CLIPSegProcessor, Dict[str, Any]], PreTrainedModel]:
            The processor and the pre-trained CLIPSeg model as a tuple.
    """
    pretrained_model_name = pretrained_model_name or "CIDAS/clipseg-rd64-refined"
    processor = CLIPSegProcessor.from_pretrained(
        pretrained_model_name, cache_dir="data/weights/clipseg", use_fast=True
    )
    model = CLIPSegForImageSegmentation.from_pretrained(
        pretrained_model_name, cache_dir="data/weights/clipseg"
    )
    return processor, model
