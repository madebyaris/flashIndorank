"""Core reranking logic.

Two public entry points:

* :func:`rerank` - score passages with a single cross-encoder.
* :func:`rerank_cascade` - the "stronger but still lightweight" path: a tiny
  model prunes the candidate set, then a stronger model re-scores only the
  survivors. This gives close-to-strong-model quality at a fraction of the cost,
  which is the whole point on a cheap VPS.

Rankers are cached as warm singletons (one ONNX session per model + max_length),
so weights are loaded from disk exactly once per process.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import onnxruntime as ort
from flashrank import Ranker, RerankRequest
from flashrank.Config import model_file_map

from .config import settings
from .models import is_supported

PassageInput = Union[str, Dict[str, Any]]

_ranker_cache: Dict[Tuple[str, int], Ranker] = {}
_cache_lock = threading.Lock()


def _apply_thread_limits(ranker: Ranker, model_name: str) -> None:
    """Rebuild the ONNX session with explicit thread counts when configured.

    FlashRank does not expose ONNX SessionOptions, so to keep CPU usage sane on
    small VPS boxes we re-create the inference session with the configured
    thread limits. Only applies to ONNX (non-listwise) rankers.
    """
    if settings.intra_op_threads <= 0 and settings.inter_op_threads <= 0:
        return
    if getattr(ranker, "session", None) is None:
        return

    so = ort.SessionOptions()
    if settings.intra_op_threads > 0:
        so.intra_op_num_threads = settings.intra_op_threads
    if settings.inter_op_threads > 0:
        so.inter_op_num_threads = settings.inter_op_threads

    model_path = str(ranker.model_dir / model_file_map[model_name])
    ranker.session = ort.InferenceSession(model_path, sess_options=so)


def get_ranker(model_name: str, max_length: Optional[int] = None) -> Ranker:
    """Return a cached, warm :class:`Ranker` for ``model_name``."""
    if not is_supported(model_name):
        raise ValueError(f"Unsupported model: {model_name!r}")

    length = max_length or settings.max_length
    key = (model_name, length)
    ranker = _ranker_cache.get(key)
    if ranker is not None:
        return ranker

    with _cache_lock:
        ranker = _ranker_cache.get(key)
        if ranker is None:
            ranker = Ranker(
                model_name=model_name,
                cache_dir=settings.cache_dir,
                max_length=length,
                log_level=settings.log_level,
            )
            _apply_thread_limits(ranker, model_name)
            _ranker_cache[key] = ranker
    return ranker


def warmup(model_names: Sequence[str], max_length: Optional[int] = None) -> None:
    """Eagerly load models so the first real request is fast."""
    for name in model_names:
        get_ranker(name, max_length=max_length)


def _normalize_passages(passages: Sequence[PassageInput]) -> List[Dict[str, Any]]:
    """Accept either plain strings or ``{id, text, meta}`` dicts.

    Returns fresh dicts so we never mutate the caller's objects (FlashRank
    sorts and adds scores in place).
    """
    normalized: List[Dict[str, Any]] = []
    for idx, passage in enumerate(passages):
        if isinstance(passage, str):
            normalized.append({"id": idx, "text": passage, "meta": {}})
        elif isinstance(passage, dict):
            if "text" not in passage:
                raise ValueError("Each passage dict must contain a 'text' field")
            item = dict(passage)
            item.setdefault("id", idx)
            item.setdefault("meta", {})
            normalized.append(item)
        else:
            raise ValueError(f"Unsupported passage type: {type(passage)!r}")
    return normalized


def _finalize(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Cast numpy scores to plain floats for clean JSON serialization."""
    for item in results:
        if "score" in item:
            item["score"] = float(item["score"])
    return results


def rerank(
    query: str,
    passages: Sequence[PassageInput],
    model_name: Optional[str] = None,
    top_k: Optional[int] = None,
    max_length: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Rerank ``passages`` against ``query`` with a single model.

    When ``model_name`` is ``None`` the configured default
    (``settings.default_model``) is used.
    """
    normalized = _normalize_passages(passages)
    if not normalized:
        return []

    ranker = get_ranker(model_name or settings.default_model, max_length=max_length)
    results = ranker.rerank(RerankRequest(query=query, passages=normalized))
    results = _finalize(results)
    if top_k is not None:
        results = results[:top_k]
    return results


def rerank_cascade(
    query: str,
    passages: Sequence[PassageInput],
    fast_model: Optional[str] = None,
    strong_model: Optional[str] = None,
    prune_to: int = 50,
    top_k: Optional[int] = None,
    max_length: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Two-stage cascade: cheap model prunes, strong model re-scores survivors.

    Stage 1 runs ``fast_model`` over every passage and keeps the top
    ``prune_to``. Stage 2 runs ``strong_model`` only over those survivors. When
    the candidate set is already small (``<= prune_to``) stage 1 is skipped
    entirely since there is nothing to prune. ``None`` resolves to the
    configured defaults (``settings.default_model`` / ``default_strong_model``).
    """
    fast_model = fast_model or settings.default_model
    strong_model = strong_model or settings.default_strong_model

    normalized = _normalize_passages(passages)
    if not normalized:
        return []

    if len(normalized) > prune_to:
        stage1 = rerank(query, normalized, model_name=fast_model, max_length=max_length)
        survivors = stage1[:prune_to]
    else:
        survivors = normalized

    results = rerank(query, survivors, model_name=strong_model, max_length=max_length)
    if top_k is not None:
        results = results[:top_k]
    return results


def timed_rerank(*args: Any, cascade: bool = False, **kwargs: Any) -> Tuple[List[Dict[str, Any]], float]:
    """Run a rerank and also return elapsed wall-clock seconds (for benchmarks)."""
    fn = rerank_cascade if cascade else rerank
    start = time.perf_counter()
    results = fn(*args, **kwargs)
    return results, time.perf_counter() - start
