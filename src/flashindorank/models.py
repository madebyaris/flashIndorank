"""Registry of supported FlashRank cross-encoder models with lightweight metadata.

Only the small, CPU-friendly ONNX cross-encoders are exposed by default; these
are the ones that make sense on a cheap VPS. The heavy listwise LLM ranker that
FlashRank also supports (``rank_zephyr_7b_v1_full``) is intentionally excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ModelInfo:
    name: str
    size_mb: float
    description: str
    multilingual: bool = False


# Ordered roughly from fastest/smallest to slowest/strongest.
MODELS: Dict[str, ModelInfo] = {
    "ms-marco-TinyBERT-L-2-v2": ModelInfo(
        name="ms-marco-TinyBERT-L-2-v2",
        size_mb=3.3,
        description="Fastest, tiniest cross-encoder. Great first-stage / cheap-VPS default.",
    ),
    "ms-marco-MiniLM-L-12-v2": ModelInfo(
        name="ms-marco-MiniLM-L-12-v2",
        size_mb=21.6,
        description="Quantized 12-layer MiniLM. Stronger relevance, still lightweight.",
    ),
    "ms-marco-MultiBERT-L-12": ModelInfo(
        name="ms-marco-MultiBERT-L-12",
        size_mb=150.0,
        description="Multilingual (100+ languages incl. Indonesian). Use for non-English corpora.",
        multilingual=True,
    ),
    "ce-esci-MiniLM-L12-v2": ModelInfo(
        name="ce-esci-MiniLM-L12-v2",
        size_mb=22.0,
        description="MiniLM fine-tuned on Amazon ESCI; good for product/e-commerce search.",
    ),
    "rank-T5-flan": ModelInfo(
        name="rank-T5-flan",
        size_mb=110.0,
        description="RankT5 (flan) encoder ranker. Strongest of the ONNX set, heaviest.",
    ),
}

# Sensible defaults for the cascade reranker: a tiny first stage that prunes
# the candidate set, and a stronger (but still small) second stage that
# re-scores only the survivors.
DEFAULT_FAST_MODEL = "ms-marco-TinyBERT-L-2-v2"
DEFAULT_STRONG_MODEL = "ms-marco-MiniLM-L-12-v2"

# The only bundled model actually trained multilingually (100+ languages incl.
# Indonesian). The English MS-MARCO models above are NOT trained on Indonesian,
# so for Bahasa Indonesia corpora this is the model to reach for. It is heavier
# (~99 MB) and only modestly better on hard Indonesian queries -- see
# benchmarks/eval_indonesian.py.
DEFAULT_MULTILINGUAL_MODEL = "ms-marco-MultiBERT-L-12"


def list_models() -> List[ModelInfo]:
    return list(MODELS.values())


def is_supported(model_name: str) -> bool:
    return model_name in MODELS
