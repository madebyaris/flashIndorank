"""Score distillation triplets with a strong teacher reranker (Margin-MSE labels).

For each (query, positive, negative) triplet we compute the teacher *margin*::

    label = teacher(query, positive) - teacher(query, negative)

following Hofstätter et al. (Margin-MSE). The student then learns to reproduce
this margin, which transfers the large teacher's ranking ability into a tiny,
VPS-friendly cross-encoder.

We score raw logits via ``transformers`` directly (not CrossEncoder.predict) so
the teacher output is unambiguous across library versions, and we de-duplicate
repeated ``(query, passage)`` pairs (the positive repeats across a query's
negatives) to cut GPU work.

Run:
    python training/teacher_score.py --teacher BAAI/bge-reranker-v2-m3 \
        --in data/distill_triplets.jsonl --out data/distill_train.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import read_jsonl, write_jsonl


def _score_pairs(teacher: str, pairs: list[tuple[str, str]], batch_size: int, max_length: int) -> list[float]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Loading teacher {teacher} on {device} ({dtype}) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(teacher)
    model = AutoModelForSequenceClassification.from_pretrained(teacher, torch_dtype=dtype).to(device).eval()

    scores: list[float] = []
    n = len(pairs)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            chunk = pairs[start : start + batch_size]
            enc = tok(
                [q for q, _ in chunk],
                [p for _, p in chunk],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits.view(-1).float().cpu().tolist()
            scores.extend(logits)
            done = start + len(chunk)
            if done % (batch_size * 50) == 0 or done == n:
                print(f"  teacher scored {done}/{n} pairs", flush=True)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default="data/distill_triplets.jsonl")
    parser.add_argument("--out", dest="out_path", default="data/distill_train.jsonl")
    parser.add_argument("--teacher", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    triplets = read_jsonl(args.in_path)
    print(f"Loaded {len(triplets)} triplets from {args.in_path}", flush=True)

    # Dedup (query, passage) pairs so the repeated positive is scored once.
    pair_index: dict[tuple[str, str], int] = {}
    pairs: list[tuple[str, str]] = []

    def _key(q: str, p: str) -> int:
        k = (q, p)
        idx = pair_index.get(k)
        if idx is None:
            idx = len(pairs)
            pair_index[k] = idx
            pairs.append(k)
        return idx

    pos_idx = [0] * len(triplets)
    neg_idx = [0] * len(triplets)
    for i, t in enumerate(triplets):
        pos_idx[i] = _key(t["query"], t["positive"])
        neg_idx[i] = _key(t["query"], t["negative"])

    print(f"Scoring {len(pairs)} unique (query, passage) pairs "
          f"(from {2 * len(triplets)} non-deduped) ...", flush=True)
    scores = _score_pairs(args.teacher, pairs, args.batch_size, args.max_length)

    rows = (
        {
            "query": t["query"],
            "positive": t["positive"],
            "negative": t["negative"],
            "label": float(scores[pos_idx[i]] - scores[neg_idx[i]]),
        }
        for i, t in enumerate(triplets)
    )
    n = write_jsonl(args.out_path, rows)
    print(f"Wrote {n} scored triplets -> {args.out_path}", flush=True)


if __name__ == "__main__":
    main()
