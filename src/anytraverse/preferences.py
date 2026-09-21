"""Helpers for working with traversability preferences."""

from anytraverse import typing as anyt


def validate_traversability_preferences(
    traversability_preferences: anyt.TraversabilityPreferences,
) -> anyt.TraversabilityPreferences:
    """
    Checks that preferences are well formed.

    Args:
        traversability_preferences (TraversabilityPreferences): The preferences to check.

    Returns:
        TraversabilityPreferences: The same preferences, unchanged.

    Raises:
        ValueError: If there are no prompts, a prompt is empty, or a weight is
            outside ``[-1, 1]``.
    """
    if not traversability_preferences:
        raise ValueError("Traversability preferences must contain at least one prompt")
    for prompt, weight in traversability_preferences.items():
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Empty prompts are not allowed in traversability preferences")
        if not -1.0 <= weight <= 1.0:
            raise ValueError(f"Weight for prompt {prompt!r} must be in [-1, 1], got {weight}")
    return traversability_preferences


def update_traversability_preferences(
    traversability_preferences: anyt.TraversabilityPreferences,
    updates: anyt.TraversabilityPreferences,
) -> anyt.TraversabilityPreferences:
    """
    Merges updates into the preferences, returning a new dict.

    Args:
        traversability_preferences (TraversabilityPreferences): The current preferences.
        updates (TraversabilityPreferences): Prompts to add or re-weight.

    Returns:
        TraversabilityPreferences: The merged preferences.
    """
    return {**traversability_preferences, **updates}


def get_prompts(traversability_preferences: anyt.TraversabilityPreferences) -> list[anyt.Prompt]:
    """Returns the prompts in insertion order."""
    return list(traversability_preferences.keys())


def get_weights(traversability_preferences: anyt.TraversabilityPreferences) -> list[anyt.Weight]:
    """Returns the weights in insertion order."""
    return list(traversability_preferences.values())


def parse_trav_pref_syntax(syntax: str) -> anyt.TraversabilityPreferences:
    """
    Parses the operator syntax ``"prompt1: weight1; prompt2: weight2"``.

    Empty segments (e.g. from a trailing ``;``) are ignored.

    Args:
        syntax (str): The text typed by the operator.

    Returns:
        TraversabilityPreferences: The parsed preferences.

    Raises:
        ValueError: If a segment is missing a ``:`` or has a non-numeric weight.

    Example:
        >>> parse_trav_pref_syntax("road: 1; bush: -0.8;")
        {'road': 1.0, 'bush': -0.8}
    """
    prefs: anyt.TraversabilityPreferences = {}
    for segment in syntax.split(";"):
        segment = segment.strip()
        if not segment:
            continue
        prompt, sep, weight = segment.partition(":")
        if not sep:
            raise ValueError(f"Segment {segment!r} is missing ':' between prompt and weight")
        prompt = prompt.strip()
        if not prompt:
            raise ValueError(f"Segment {segment!r} has an empty prompt")
        try:
            prefs[prompt] = float(weight)
        except ValueError as exc:
            raise ValueError(f"Segment {segment!r} has a non-numeric weight") from exc
    return prefs


__all__ = [
    "get_prompts",
    "get_weights",
    "parse_trav_pref_syntax",
    "update_traversability_preferences",
    "validate_traversability_preferences",
]
