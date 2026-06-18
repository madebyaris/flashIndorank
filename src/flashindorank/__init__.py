"""flashIndorank - lightweight, high-performance reranker built on FlashRank."""

from .config import settings
from .models import (
    DEFAULT_FAST_MODEL,
    DEFAULT_MULTILINGUAL_MODEL,
    DEFAULT_STRONG_MODEL,
    MODELS,
    ModelInfo,
    is_supported,
    list_models,
)
from .reranker import get_ranker, rerank, rerank_cascade, timed_rerank, warmup

__version__ = "0.1.0"

__all__ = [
    "settings",
    "MODELS",
    "ModelInfo",
    "DEFAULT_FAST_MODEL",
    "DEFAULT_STRONG_MODEL",
    "DEFAULT_MULTILINGUAL_MODEL",
    "is_supported",
    "list_models",
    "get_ranker",
    "rerank",
    "rerank_cascade",
    "timed_rerank",
    "warmup",
    "__version__",
]
