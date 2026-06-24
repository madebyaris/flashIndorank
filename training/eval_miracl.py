"""Evaluate rerankers on MIRACL-id with the official retrieve-then-rerank protocol.

1. BM25 retrieves top-``--retrieve-k`` passages from the MIRACL-id corpus.
2. A cross-encoder reranker re-scores the candidate list.
3. Metrics are computed with ``pytrec_eval`` (nDCG@10, MRR@10, Recall@k).

BM25 hits are cached under ``data/miracl/`` so repeated eval runs skip indexing.

Run:

    python training/eval_miracl.py --model models/ft-id-ce-v2
    python training/eval_miracl.py --model cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Sequence

from common import miracl_metrics, read_jsonl, write_jsonl
from miracl_data import MiraclBM25, load_qrels, load_topics, qrels_path, topics_path

warnings.filterwarnings("ignore")

FLASHRANK_NAMES = {
    "ms-marco-TinyBERT-L-2-v2",
    "ms-marco-MiniLM-L-12-v2",
    "ms-marco-MultiBERT-L-12",
    "ce-esci-MiniLM-L12-v2",
    "rank-T5-flan",
}


def _cache_path(cache: str, split: str, retrieve_k: int) -> Path:
    return Path(cache) / f"bm25_{split}_top{retrieve_k}.jsonl"


def _build_or_load_bm25_hits(
    split: str,
    retrieve_k: int,
    cache: str,
    max_corpus_docs: int,
    max_queries: int,
) -> List[dict]:
    cache_file = _cache_path(cache, split, retrieve_k)
    if cache_file.exists():
        return read_jsonl(cache_file)

    topics = load_topics(topics_path(split, cache))
    qrels = load_qrels(qrels_path(split, cache))
    qids = list(topics.keys())
    if max_queries:
        qids = qids[:max_queries]

    print(f"Building BM25 index for MIRACL-id ({max_corpus_docs or 'full'} docs)...")
    bm25 = MiraclBM25.from_cache(cache=cache, max_docs=max_corpus_docs)

    rows = []
    for i, qid in enumerate(qids):
        rel_ids = set(qrels.get(qid, {}))
        hits = bm25.search(topics[qid], k=retrieve_k, exclude=set())
        rows.append(
            {
                "query_id": qid,
                "query": topics[qid],
                "candidates": [{"docid": d, "text": t} for d, t, _ in hits],
                "relevant_docids": sorted(rel_ids),
            }
        )
        if (i + 1) % 100 == 0:
            print(f"  BM25 {i + 1}/{len(qids)}")

    write_jsonl(cache_file, rows)
    print(f"Cached BM25 hits -> {cache_file}")
    return rows


def _rerank_crossencoder(model_path: str, items: Sequence[dict], max_length: int) -> Dict[str, List[str]]:
    import torch
    from sentence_transformers import CrossEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 128 if device == "cuda" else 32
    model = CrossEncoder(model_path, max_length=max_length, device=device)
    ranked: Dict[str, List[str]] = {}
    for it in items:
        pairs = [[it["query"], c["text"]] for c in it["candidates"]]
        scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        ranked[it["query_id"]] = [it["candidates"][i]["docid"] for i in order]
    return ranked


def _rerank_flashrank(model_name: str, items: Sequence[dict]) -> Dict[str, List[str]]:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from flashindorank import rerank

    ranked: Dict[str, List[str]] = {}
    for it in items:
        passages = [{"id": c["docid"], "text": c["text"]} for c in it["candidates"]]
        results = rerank(it["query"], passages, model_name=model_name)
        ranked[it["query_id"]] = [str(r["id"]) for r in results]
    return ranked


def _metrics(
    qrels: Dict[str, Dict[str, int]],
    run: Dict[str, List[str]],
    k: int,
    recall_k: int,
) -> Dict[str, float]:
    return miracl_metrics(qrels, run, k=k, recall_k=recall_k)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="CrossEncoder path/name or FlashRank model")
    parser.add_argument("--split", default="dev", choices=["dev", "train"])
    parser.add_argument("--miracl-cache", default="data/miracl")
    parser.add_argument("--retrieve-k", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--max-corpus-docs", type=int, default=0)
    parser.add_argument("--max-queries", type=int, default=0, help="0 = all queries in split")
    parser.add_argument("--metrics-k", type=int, default=10)
    args = parser.parse_args()

    items = _build_or_load_bm25_hits(
        args.split,
        args.retrieve_k,
        args.miracl_cache,
        args.max_corpus_docs,
        args.max_queries,
    )
    qrels = load_qrels(qrels_path(args.split, args.miracl_cache))
    qrels = {it["query_id"]: {did: 1 for did in it["relevant_docids"]} for it in items}

    print(f"Reranking {len(items)} MIRACL-id {args.split} queries "
          f"(BM25 top-{args.retrieve_k}) with {args.model!r}...")

    if args.model in FLASHRANK_NAMES:
        run = _rerank_flashrank(args.model, items)
    else:
        run = _rerank_crossencoder(args.model, items, args.max_length)

    metrics = _metrics(qrels, run, k=args.metrics_k, recall_k=args.retrieve_k)
    print(f"\nMIRACL-id {args.split} (BM25 top-{args.retrieve_k} -> rerank):\n")
    for name, value in metrics.items():
        print(f"  {name:<12} {value:.4f}")

    out = Path(args.miracl_cache) / f"eval_{args.split}_{Path(args.model).name}.json"
    out.write_text(json.dumps({"model": args.model, "metrics": metrics}, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
