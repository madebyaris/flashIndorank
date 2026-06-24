"""Convert labeled (query, passage, label) pairs into (query, positive, negative)
triplets for Margin-MSE distillation.

The in-domain TyDi+MIRACL data (``prepare_data.py`` output) is in pointwise BCE
form. Grouping by query lets us reuse the same mined hard negatives as
distillation triplets, which the teacher then scores.

Run:
    python training/pairs_to_triplets.py --in data/train.jsonl \
        --out data/indomain_triplets.jsonl --negs-per-query 6
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

from common import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default="data/train.jsonl")
    parser.add_argument("--out", dest="out_path", default="data/indomain_triplets.jsonl")
    parser.add_argument("--negs-per-query", type=int, default=6)
    parser.add_argument("--max-chars", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = read_jsonl(args.in_path)

    pos: dict[str, list[str]] = defaultdict(list)
    neg: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        q = (r.get("query") or "").strip()
        p = (r.get("passage") or "").strip()[: args.max_chars]
        if not q or not p:
            continue
        label = float(r.get("label", 0.0))
        (pos if label >= 0.5 else neg)[q].append(p)

    triplets = []
    for q, positives in pos.items():
        negatives = neg.get(q) or []
        if not negatives:
            continue
        sampled = (
            negatives
            if len(negatives) <= args.negs_per_query
            else rng.sample(negatives, args.negs_per_query)
        )
        for positive in positives:
            for negative in sampled:
                triplets.append({"query": q, "positive": positive, "negative": negative})

    n = write_jsonl(args.out_path, triplets)
    print(
        f"{len(pos)} queries with positives -> {n} triplets "
        f"(negs/query<= {args.negs_per_query}) -> {args.out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
