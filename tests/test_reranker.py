"""Unit/integration tests for the core reranker.

These download small ONNX models on first run (a few MB) and require network
access the first time only.
"""

import pytest

from flashindorank import is_supported, list_models, rerank, rerank_cascade
from flashindorank.reranker import _normalize_passages

QUERY = "Which animal is a giant panda?"
PASSAGES = [
    {"id": "a", "text": "The giant panda is a bear species native to China."},
    {"id": "b", "text": "Python is a high-level programming language."},
    {"id": "c", "text": "Pandas eat mostly bamboo in the wild."},
]


def test_model_registry():
    names = [m.name for m in list_models()]
    assert "ms-marco-TinyBERT-L-2-v2" in names
    assert "ms-marco-MiniLM-L-12-v2" in names
    assert is_supported("ms-marco-TinyBERT-L-2-v2")
    assert not is_supported("does-not-exist")


def test_normalize_accepts_strings_and_dicts():
    out = _normalize_passages(["hello", {"text": "world", "id": 9}])
    assert out[0]["text"] == "hello"
    assert out[0]["id"] == 0
    assert out[1]["id"] == 9
    assert "meta" in out[0]


def test_normalize_rejects_missing_text():
    with pytest.raises(ValueError):
        _normalize_passages([{"id": 1}])


def test_rerank_orders_relevant_first():
    results = rerank(QUERY, PASSAGES)
    assert len(results) == 3
    # The unrelated programming passage must not be ranked first.
    assert results[0]["id"] in {"a", "c"}
    assert results[0]["score"] >= results[-1]["score"]
    assert isinstance(results[0]["score"], float)


def test_rerank_top_k():
    results = rerank(QUERY, PASSAGES, top_k=1)
    assert len(results) == 1


def test_rerank_empty():
    assert rerank(QUERY, []) == []


def test_cascade_matches_single_when_small():
    # With fewer passages than prune_to, stage 1 is skipped; the top result
    # should still be a panda passage.
    results = rerank_cascade(QUERY, PASSAGES, prune_to=50, top_k=2)
    assert len(results) == 2
    assert results[0]["id"] in {"a", "c"}


def test_cascade_prunes_large_set():
    passages = PASSAGES + [
        {"id": f"noise-{i}", "text": f"Unrelated sentence number {i}."} for i in range(30)
    ]
    results = rerank_cascade(QUERY, passages, prune_to=5, top_k=3)
    assert len(results) == 3
    assert results[0]["id"] in {"a", "c"}
