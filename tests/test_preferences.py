import pytest

from anytraverse.preferences import (
    get_prompts,
    get_weights,
    parse_trav_pref_syntax,
    update_traversability_preferences,
    validate_traversability_preferences,
)


class TestParseTravPrefSyntax:
    def test_basic(self) -> None:
        assert parse_trav_pref_syntax("road: 1; bush: -0.8") == {"road": 1.0, "bush": -0.8}

    @pytest.mark.parametrize("text", ["road: 1;", " road : 1 ; ; ", "road:1"])
    def test_ignores_empty_segments_and_whitespace(self, text: str) -> None:
        assert parse_trav_pref_syntax(text) == {"road": 1.0}

    def test_empty_string_gives_empty_dict(self) -> None:
        assert parse_trav_pref_syntax("") == {}
        assert parse_trav_pref_syntax("  ;  ") == {}

    def test_later_duplicate_wins(self) -> None:
        assert parse_trav_pref_syntax("road: 1; road: 0.5") == {"road": 0.5}

    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ("road 1", "missing ':'"),
            (": 1", "empty prompt"),
            ("road: high", "non-numeric weight"),
        ],
    )
    def test_invalid_segments_raise(self, text: str, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            parse_trav_pref_syntax(text)


class TestValidate:
    def test_returns_input(self, prefs) -> None:
        assert validate_traversability_preferences(prefs) is prefs

    @pytest.mark.parametrize(
        ("bad", "message"),
        [
            ({}, "at least one prompt"),
            ({"": 1.0}, "Empty prompts"),
            ({"  ": 1.0}, "Empty prompts"),
            ({"road": 1.5}, "must be in \\[-1, 1\\]"),
            ({"road": -2.0}, "must be in \\[-1, 1\\]"),
        ],
    )
    def test_rejects_bad_preferences(self, bad, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            validate_traversability_preferences(bad)


def test_update_returns_new_merged_dict(prefs) -> None:
    updated = update_traversability_preferences(prefs, {"bush": 0.0, "mud": -0.5})
    assert updated == {"road": 1.0, "bush": 0.0, "rock": 0.45, "mud": -0.5}
    assert prefs["bush"] == -0.8


def test_get_prompts_and_weights_preserve_order(prefs) -> None:
    assert get_prompts(prefs) == ["road", "bush", "rock"]
    assert get_weights(prefs) == [1.0, -0.8, 0.45]
