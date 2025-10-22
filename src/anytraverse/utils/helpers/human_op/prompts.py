from anytraverse.config.utils import WeightedPrompt
from typing import List, Dict


Prompts = List[WeightedPrompt]
PromptsDict = Dict[str, float]


def update_prompts(prompts: Prompts, delta_prompts: Prompts) -> Prompts:
    """
    Updates the `prompts` according to the items in `delta_prompts`:
    1. Any prompt texts common to both, will have the value as declared in `delta_prompts`.
    2. Any prompt texts in `delta_prompts`, but not in `prompts`, will appear as a new prompt
        in the output.

    Args:
        prompts (Prompts): The prompts to update.
        delta_prompts (Prompts): The updates to make.

    Return:
        Prompts: The updated prompts.
    """
    prompts_dict: PromptsDict = dict(prompts)
    delta_prompts_dict: PromptsDict = dict(delta_prompts)
    prompts_dict.update(delta_prompts_dict)
    return list(prompts_dict.items())
