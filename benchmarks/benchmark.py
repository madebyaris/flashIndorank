"""Latency/throughput benchmark for the reranker models and the cascade.

Run with the venv active:

    python benchmarks/benchmark.py --passages 100 --runs 20

Designed to show that even the strong path stays fast on modest CPUs.
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable, List

from flashindorank import (
    DEFAULT_FAST_MODEL,
    DEFAULT_STRONG_MODEL,
    get_ranker,
    rerank,
    rerank_cascade,
)

QUERY = "What is the capital city of Indonesia?"

CORPUS = [
    "Jakarta is the capital and largest city of Indonesia.",
    "Indonesia is a country in Southeast Asia and Oceania.",
    "Nusantara is the planned future capital of Indonesia in Kalimantan.",
    "Bali is a popular tourist island in Indonesia.",
    "The Python programming language is widely used for data science.",
    "Mount Bromo is an active volcano in East Java.",
    "Rendang is a spicy meat dish originating from Indonesia.",
    "The Garuda is the national symbol of Indonesia.",
    "Bandung is the capital of the West Java province.",
    "Surabaya is the second largest city in Indonesia.",
]


def _make_passages(n: int) -> List[dict]:
    passages = []
    for i in range(n):
        text = CORPUS[i % len(CORPUS)]
        passages.append({"id": i, "text": f"{text} (variant {i})"})
    return passages


def _bench(label: str, fn: Callable[[], list], runs: int) -> None:
    # warmup
    fn()
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - start) * 1000)
    p50 = statistics.median(timings)
    p95 = sorted(timings)[max(0, int(len(timings) * 0.95) - 1)]
    qps = 1000 / p50 if p50 else float("inf")
    print(f"{label:<46} p50={p50:7.2f}ms  p95={p95:7.2f}ms  ~{qps:6.1f} req/s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--passages", type=int, default=100)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--prune-to", type=int, default=20)
    args = parser.parse_args()

    passages = _make_passages(args.passages)
    print(f"Benchmark: query x {len(passages)} passages, {args.runs} runs each\n")

    # Preload so model-download / load time is excluded from timings.
    get_ranker(DEFAULT_FAST_MODEL)
    get_ranker(DEFAULT_STRONG_MODEL)

    _bench(f"single  {DEFAULT_FAST_MODEL}", lambda: rerank(QUERY, passages, model_name=DEFAULT_FAST_MODEL), args.runs)
    _bench(f"single  {DEFAULT_STRONG_MODEL}", lambda: rerank(QUERY, passages, model_name=DEFAULT_STRONG_MODEL), args.runs)
    _bench(
        f"cascade Tiny->MiniLM (prune {args.prune_to})",
        lambda: rerank_cascade(QUERY, passages, prune_to=args.prune_to),
        args.runs,
    )


if __name__ == "__main__":
    main()
