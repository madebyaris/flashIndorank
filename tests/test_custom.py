"""Tests for the custom local ONNX reranker path.

The end-to-end serving test needs an exported model directory; it is skipped
unless one is available (set FLASHINDORANK_TEST_CUSTOM_MODEL, or have
models/ft-id-ce-onnx present from the training pipeline).
"""

import os
from pathlib import Path

import pytest

import flashindorank
from flashindorank import config, reranker


def _candidate_model_dir():
    env = os.environ.get("FLASHINDORANK_TEST_CUSTOM_MODEL")
    if env and Path(env, "model.onnx").exists():
        return env
    default = Path(__file__).resolve().parent.parent / "models" / "ft-id-ce-onnx"
    if (default / "model.onnx").exists():
        return str(default)
    return None


def test_is_known_model_custom_registration(monkeypatch):
    monkeypatch.setitem(config.settings.custom_models, "my-custom", "/some/path")
    try:
        assert reranker.is_known_model("my-custom")
        assert not reranker.is_known_model("totally-unknown-model")
    finally:
        config.settings.custom_models.pop("my-custom", None)


def test_custom_reranker_missing_dir_raises(tmp_path):
    from flashindorank import CustomReranker

    with pytest.raises(FileNotFoundError):
        CustomReranker(str(tmp_path))  # no model.onnx


@pytest.mark.skipif(_candidate_model_dir() is None, reason="no exported ONNX model available")
def test_custom_onnx_reranks_indonesian(monkeypatch):
    model_dir = _candidate_model_dir()
    monkeypatch.setitem(config.settings.custom_models, "id-test", model_dir)
    reranker._ranker_cache.clear()

    query = "Bagaimana cara menurunkan berat badan?"
    passages = [
        {"id": "rel", "text": "Olahraga teratur dan pola makan sehat membantu mengurangi bobot tubuh."},
        {"id": "n1", "text": "Mobil listrik semakin populer di kota-kota besar dunia."},
        {"id": "n2", "text": "Harga emas global naik tajam dalam sepekan terakhir."},
    ]
    results = flashindorank.rerank(query, passages, model_name="id-test")
    assert results[0]["id"] == "rel"
    assert isinstance(results[0]["score"], float)
    config.settings.custom_models.pop("id-test", None)
    reranker._ranker_cache.clear()
