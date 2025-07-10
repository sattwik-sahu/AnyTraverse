from anytraverse import typing as anyt


def update_traversability_preferences(
    traversability_preferences: anyt.TraversabilityPreferences,
    updates: anyt.TraversabilityPreferences,
) -> anyt.TraversabilityPreferences:
    """
    Updates the traversability preferences using the given udpates.

    Args:
        prefs (TraversabilityPreferences): The traversability preferences to update.
        updates (TraversabilityPreferences): The updates.

    Returns:
        TraversabilityPreferences:
            The updated traversability preferences.
    """
    return {**traversability_preferences, **updates}


def get_prompts(
    traversability_preferences: anyt.TraversabilityPreferences,
) -> list[anyt.Prompt]:
    return list(traversability_preferences.keys())


def get_weights(
    traversability_preferences: anyt.TraversabilityPreferences,
) -> list[anyt.Weight]:
    return list(traversability_preferences.values())


def parse_trav_pref_syntax(syntax: str) -> anyt.TraversabilityPreferences:
    """
    Parses traversability preference sytax to obtain a
    `TraversabilityPreferences` type `dict`.

    The syntax is:
    `prompt1: weight1; prompt2: weight2; ... ;`

    Args:
        syntax (str): The syntax describing the traversability preferences.

        Returns:
            TraversabilityPreferences:
                The traversability preferences described in the syntax as a `dict[str, float]`
    """
    syntax = syntax.strip()
    pws = syntax.split(";")
    prefs: anyt.TraversabilityPreferences = dict()
    for pw in pws:
        prompt, weight = pw.split(":")
        weight = float(weight)
        prefs[prompt.strip()] = weight
    return prefs
