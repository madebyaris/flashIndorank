"""Evaluate rerankers on the held-out Indonesian eval set.

Supports two kinds of models:
* a local / Hub sentence-transformers CrossEncoder (path or name), and
* a bundled FlashRank model name (routed through flashindorank.rerank),

so you can compare the English default, the multilingual base, and your
fine-tuned model on the exact same Indonesian queries.

Run:
    python training/evaluate.py --models \
        ms-marco-MiniLM-L-12-v2 \
        cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 \
        models/ft-id-ce
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

from common import aggregate, read_jsonl

warnings.filterwarnings("ignore")

# bundled FlashRank model names are handled via flashindorank
FLASHRANK_NAMES = {
    "ms-marco-TinyBERT-L-2-v2",
    "ms-marco-MiniLM-L-12-v2",
    "ms-marco-MultiBERT-L-12",
    "ce-esci-MiniLM-L12-v2",
    "rank-T5-flan",
}


def _rank_of_positive_flashrank(model_name, items):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from flashindorank import rerank

    ranks = []
    for it in items:
        passages = [{"id": "POS", "text": it["positive"]}] + [
            {"id": i, "text": neg} for i, neg in enumerate(it["negatives"])
        ]
        results = rerank(it["query"], passages, model_name=model_name)
        order = [r["id"] for r in results]
        ranks.append(order.index("POS") + 1)
    return ranks


def _rank_of_positive_crossencoder(model_path, items, max_length):
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_path, max_length=max_length)
    ranks = []
    for it in items:
        candidates = [it["positive"]] + list(it["negatives"])  # index 0 == positive
        scores = model.predict([[it["query"], c] for c in candidates])
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        ranks.append(order.index(0) + 1)
    return ranks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", default="data/eval.jsonl")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    items = read_jsonl(args.eval_file)
    print(f"Evaluating on {len(items)} Indonesian eval items "
          f"(1 positive + {len(items[0]['negatives'])} negatives each)\n")

    header = f"{'model':<46}{'top1':>8}{'MRR':>8}{f'nDCG@{args.k}':>10}"
    print(header)
    print("-" * len(header))
    for model in args.models:
        if model in FLASHRANK_NAMES:
            ranks = _rank_of_positive_flashrank(model, items)
        else:
            ranks = _rank_of_positive_crossencoder(model, items, args.max_length)
        m = aggregate(ranks, k=args.k)
        print(f"{model:<46}{m['top1_acc']:>8.3f}{m['mrr']:>8.3f}{m[f'ndcg@{args.k}']:>10.3f}")


if __name__ == "__main__":
    main()
