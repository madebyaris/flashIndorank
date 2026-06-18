"""Build Indonesian cross-encoder train/eval data from TyDi QA (Gold Passage).

TyDi QA is a parquet-native, human-annotated multilingual QA dataset; we keep
only the Indonesian rows. Each row gives a question and its gold passage
(a positive). We mine *hard* negatives by lexical overlap from the passage pool.

Outputs (under --out-dir):
* train.jsonl - {"query", "passage", "label"} pairs (1 positive + N negatives)
* eval.jsonl  - {"query", "positive", "negatives": [...]} ranking items

Run (venv with training deps active):
    python training/prepare_data.py --train-queries 800 --eval-queries 200
"""

from __future__ import annotations

import argparse
import random
import warnings
from pathlib import Path

from common import LexicalIndex, write_jsonl

warnings.filterwarnings("ignore")


def _load_indonesian_rows():
    from datasets import load_dataset

    rows = []
    seen_passages = {}
    for split in ("train", "validation"):
        ds = load_dataset("tydiqa", "secondary_task", split=split)
        ds = ds.filter(lambda r: r["id"].startswith("indonesian"))
        for r in ds:
            q = r["question"].strip()
            ctx = r["context"].strip()
            if not q or not ctx:
                continue
            if ctx not in seen_passages:
                seen_passages[ctx] = len(seen_passages)
            rows.append({"query": q, "passage": ctx})
    return rows, list(seen_passages.keys())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--train-queries", type=int, default=800)
    parser.add_argument("--eval-queries", type=int, default=200)
    parser.add_argument("--train-negatives", type=int, default=3, help="negatives per train query")
    parser.add_argument("--eval-negatives", type=int, default=9, help="negatives per eval query")
    parser.add_argument("--hard-frac", type=float, default=0.7, help="fraction of negatives that are hard")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    print("Loading Indonesian TyDi QA rows...")
    rows, passages = _load_indonesian_rows()
    print(f"  {len(rows)} (query, gold passage) rows, {len(passages)} unique passages")

    passage_to_idx = {p: i for i, p in enumerate(passages)}
    index = LexicalIndex(passages)

    random.shuffle(rows)
    need = args.train_queries + args.eval_queries
    rows = rows[:need]
    train_rows = rows[: args.train_queries]
    eval_rows = rows[args.train_queries : args.train_queries + args.eval_queries]

    def sample_negatives(query: str, gold_idx: int, n: int):
        n_hard = int(round(n * args.hard_frac))
        hard = [i for i in index.top_overlap(query, k=n_hard * 3, exclude_idx=gold_idx)][:n_hard]
        negs = list(hard)
        while len(negs) < n:
            j = random.randrange(len(passages))
            if j != gold_idx and j not in negs:
                negs.append(j)
        return [passages[i] for i in negs[:n]]

    # ---- training pairs ----
    train_pairs = []
    for r in train_rows:
        gold_idx = passage_to_idx[r["passage"]]
        train_pairs.append({"query": r["query"], "passage": r["passage"], "label": 1.0})
        for neg in sample_negatives(r["query"], gold_idx, args.train_negatives):
            train_pairs.append({"query": r["query"], "passage": neg, "label": 0.0})
    random.shuffle(train_pairs)
    n_train = write_jsonl(Path(args.out_dir) / "train.jsonl", train_pairs)

    # ---- eval ranking items ----
    eval_items = []
    for r in eval_rows:
        gold_idx = passage_to_idx[r["passage"]]
        negs = sample_negatives(r["query"], gold_idx, args.eval_negatives)
        eval_items.append({"query": r["query"], "positive": r["passage"], "negatives": negs})
    n_eval = write_jsonl(Path(args.out_dir) / "eval.jsonl", eval_items)

    print(f"Wrote {n_train} train pairs -> {args.out_dir}/train.jsonl")
    print(f"Wrote {n_eval} eval items  -> {args.out_dir}/eval.jsonl")


if __name__ == "__main__":
    main()
