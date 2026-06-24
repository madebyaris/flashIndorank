"""Build (query, positive, negative) triplets for Margin-MSE distillation.

Primary source: **mMARCO Indonesian** training data
(``crystina-z/mmarco-passage`` → ``indonesian-train.jsonl.gz``), which ships each
query with one positive passage and ~30 pre-mined BM25 hard negatives, all with
inline text. This gives large-scale Indonesian ranking signal **without** loading
the 8.8M-passage collection or building BM25 ourselves.

Output (under --out-dir):
* ``distill_triplets.jsonl``  - {"query", "positive", "negative"} (one per neg)
* ``distill_eval.jsonl``      - held-out {"query", "positives", "negatives"} for
                                a CrossEncoderRerankingEvaluator during training

The teacher scores these triplets next (``teacher_score.py``); training uses the
teacher margins (``train_distill.py``).

Run:
    python training/prepare_distill_data.py --mmarco-queries 80000 --negs-per-query 5
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path

from common import write_jsonl

MMARCO_REPO = "crystina-z/mmarco-passage"
MMARCO_FILE = "indonesian-train.jsonl.gz"


def _download_mmarco(cache: str) -> Path:
    from huggingface_hub import hf_hub_download

    print(f"Downloading {MMARCO_REPO}/{MMARCO_FILE} (~1.6 GB, cached) ...", flush=True)
    path = hf_hub_download(
        repo_id=MMARCO_REPO,
        filename=MMARCO_FILE,
        repo_type="dataset",
        cache_dir=cache or None,
    )
    return Path(path)


def _first_text(passages: list[dict]) -> str | None:
    for p in passages:
        t = (p.get("text") or "").strip()
        if t:
            return t
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--hf-cache", default="", help="HF datasets cache dir (defaults to HF_HOME)")
    parser.add_argument("--mmarco-queries", type=int, default=80000, help="queries to use for training")
    parser.add_argument("--negs-per-query", type=int, default=5, help="hard negatives sampled per query")
    parser.add_argument("--eval-queries", type=int, default=500, help="held-out queries for rerank eval")
    parser.add_argument("--max-chars", type=int, default=2000, help="truncate passage text to N chars")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    src = _download_mmarco(args.hf_cache)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_needed = args.mmarco_queries + args.eval_queries
    eval_items: list[dict] = []
    n_train_triplets = 0
    n_query = 0
    clip = args.max_chars

    train_path = out_dir / "distill_triplets.jsonl"
    train_path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(src, "rt", encoding="utf-8") as fin, open(train_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if n_query >= total_needed:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            q = row.get("query")
            query = (q[0] if isinstance(q, list) and q else q) or ""
            query = str(query).strip()
            positive = _first_text(row.get("positive_passages") or [])
            negs = [
                (p.get("text") or "").strip()
                for p in (row.get("negative_passages") or [])
                if (p.get("text") or "").strip()
            ]
            if not query or not positive or not negs:
                continue

            positive = positive[:clip]
            negs = [n[:clip] for n in negs]

            # First eval_queries rows -> rerank evaluator (keep ALL negatives).
            if len(eval_items) < args.eval_queries:
                eval_items.append(
                    {"query": query, "positives": [positive], "negatives": negs}
                )
                n_query += 1
                continue

            sampled = negs if len(negs) <= args.negs_per_query else rng.sample(negs, args.negs_per_query)
            for neg in sampled:
                fout.write(
                    json.dumps({"query": query, "positive": positive, "negative": neg}, ensure_ascii=False)
                    + "\n"
                )
                n_train_triplets += 1
            n_query += 1

            if n_query % 10000 == 0:
                print(f"  processed {n_query} queries, {n_train_triplets} triplets ...", flush=True)

    write_jsonl(out_dir / "distill_eval.jsonl", eval_items)

    print(
        f"Done: {n_train_triplets} train triplets ({n_query - len(eval_items)} queries) "
        f"-> {train_path}",
        flush=True,
    )
    print(f"      {len(eval_items)} eval queries -> {out_dir / 'distill_eval.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
