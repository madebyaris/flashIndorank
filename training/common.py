"""Shared helpers for the Indonesian fine-tuning pipeline.

Kept dependency-light (stdlib only) so it can be imported by data prep,
training, and evaluation without pulling anything heavy.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def read_jsonl(path: str | Path) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


class LexicalIndex:
    """Tiny TF-based lexical scorer used to mine *hard* negatives.

    Hard negatives (passages that share words with the query but are not the
    gold passage) force the cross-encoder to learn real relevance rather than
    trivial topic mismatch, which is what makes fine-tuning actually move the
    needle.
    """

    def __init__(self, passages: Sequence[str]):
        self.passages = list(passages)
        self.toks = [Counter(tokenize(p)) for p in self.passages]

    def top_overlap(self, query: str, k: int, exclude_idx: int) -> List[int]:
        q = Counter(tokenize(query))
        scored = []
        for i, tf in enumerate(self.toks):
            if i == exclude_idx:
                continue
            score = sum(min(q[t], tf[t]) for t in q)
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)
        return [i for _score, i in scored[:k]]


def mrr(rank: int) -> float:
    return 1.0 / rank if rank > 0 else 0.0


def ndcg_at_k(rank: int, k: int = 10) -> float:
    """nDCG for a single relevant item at position ``rank`` (1-indexed)."""
    if rank == 0 or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)  # IDCG == 1 for a single relevant doc


def ndcg_at_k_multi(relevances: List[int], k: int = 10) -> float:
    """nDCG@k for a ranked list where ``relevances[i]`` is the gain at rank i."""
    dcg = sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))
    ideal = sorted(relevances, reverse=True)
    idcg = sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def miracl_metrics(
    qrels: Dict[str, Dict[str, int]],
    run: Dict[str, List[str]],
    k: int = 10,
    recall_k: int = 100,
) -> Dict[str, float]:
    """Compute mean nDCG@k, MRR@k, and Recall@recall_k over labeled queries."""
    ndcgs: List[float] = []
    mrrs: List[float] = []
    recalls: List[float] = []
    for qid, rel_docs in qrels.items():
        ranked = run.get(qid, [])
        rel_set = set(rel_docs)
        if not rel_set:
            continue
        gains = [1 if docid in rel_set else 0 for docid in ranked[:k]]
        ndcgs.append(ndcg_at_k_multi(gains, k=k))
        rr = 0.0
        for i, docid in enumerate(ranked[:k], start=1):
            if docid in rel_set:
                rr = 1.0 / i
                break
        mrrs.append(rr)
        found = sum(1 for docid in ranked[:recall_k] if docid in rel_set)
        recalls.append(found / len(rel_set))
    n = len(ndcgs) or 1
    return {
        f"ndcg@{k}": sum(ndcgs) / n,
        f"mrr@{k}": sum(mrrs) / n,
        f"recall@{recall_k}": sum(recalls) / n,
        "n": len(ndcgs),
    }


def aggregate(ranks: List[int], k: int = 10) -> Dict[str, float]:
    n = len(ranks) or 1
    return {
        "top1_acc": sum(1 for r in ranks if r == 1) / n,
        "mrr": sum(mrr(r) for r in ranks) / n,
        f"ndcg@{k}": sum(ndcg_at_k(r, k) for r in ranks) / n,
        "n": len(ranks),
    }
