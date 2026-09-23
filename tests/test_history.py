import pytest
import torch

from anytraverse.history import EncodingHistory
from anytraverse.interfaces import History
from tests.conftest import unit


@pytest.fixture
def history() -> EncodingHistory:
    return EncodingHistory(similarity_func=torch.cosine_similarity)


def test_empty_history_raises(history: EncodingHistory) -> None:
    assert len(history) == 0
    with pytest.raises(ValueError, match="empty"):
        history.find_best_match(unit(1.0))
    with pytest.raises(ValueError, match="empty"):
        history.encodings


def test_add_stacks_encodings_and_copies_preferences(history: EncodingHistory) -> None:
    prefs = {"road": 1.0}
    key, stored = history.add(unit(1.0), prefs)
    prefs["road"] = 0.0
    assert stored == {"road": 1.0}
    assert key.shape == (1, 4)
    history.add(unit(0.0, 1.0), {"bush": -1.0})
    assert history.encodings.shape == (2, 4)
    assert len(history) == 2


def test_find_best_match_returns_element_and_similarity(history: EncodingHistory) -> None:
    history.add(unit(1.0), {"a": 1.0})
    history.add(unit(0.0, 1.0), {"b": 1.0})
    history.add(unit(0.0, 0.0, 1.0), {"c": 1.0})

    (encoding, prefs), similarity = history.find_best_match(unit(0.1, 0.9))
    assert prefs == {"b": 1.0}
    torch.testing.assert_close(encoding, unit(0.0, 1.0))
    assert similarity == pytest.approx(
        torch.cosine_similarity(unit(0.1, 0.9), unit(0.0, 1.0)).item()
    )


def test_find_best_match_accepts_flat_query(history: EncodingHistory) -> None:
    history.add(unit(1.0), {"a": 1.0})
    (_, prefs), _ = history.find_best_match(unit(1.0).ravel())
    assert prefs == {"a": 1.0}


def test_vectorized_matches_naive(history: EncodingHistory) -> None:
    keys = torch.randn(10, 4)
    for i, k in enumerate(keys):
        history.add(k, {str(i): 1.0})
    query = torch.randn(1, 4)
    naive = max(range(10), key=lambda i: torch.cosine_similarity(query, keys[i : i + 1]).item())
    (_, prefs), _ = history.find_best_match(query)
    assert prefs == {str(naive): 1.0}


def test_max_size_evicts_oldest() -> None:
    history = EncodingHistory(torch.cosine_similarity, max_size=2)
    history.add(unit(1.0), {"a": 1.0})
    history.add(unit(0.0, 1.0), {"b": 1.0})
    history.add(unit(0.0, 0.0, 1.0), {"c": 1.0})
    assert len(history) == 2
    assert history.encodings.shape == (2, 4)
    assert [p for _, p in history.store] == [{"b": 1.0}, {"c": 1.0}]
    (_, prefs), _ = history.find_best_match(unit(1.0))
    assert prefs != {"a": 1.0}


def test_invalid_max_size() -> None:
    with pytest.raises(ValueError, match="max_size"):
        EncodingHistory(torch.cosine_similarity, max_size=0)


def test_base_history_is_abstract() -> None:
    with pytest.raises(TypeError):
        History()  # type: ignore[abstract]
