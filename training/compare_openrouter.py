"""Compare our fine-tuned Indonesian reranker against OpenRouter rerank models.

Evaluates on a small slice (default 15 queries, to keep API cost low) of the
held-out Indonesian eval set. For each query we build [positive] + hard
negatives and measure where each reranker places the positive.

Metrics: top-1 accuracy, MRR, nDCG@10 (single relevant doc per query).

Auth: set OPENROUTER_API_KEY in the environment.

Run:
    python training/compare_openrouter.py --n 15 \
        --onnx-dir models/ft-id-ce-onnx \
        --openrouter-models cohere/rerank-v3.5 nvidia/llama-nemotron-rerank-vl-1b-v2:free
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import aggregate, read_jsonl  # noqa: E402

warnings.filterwarnings("ignore")

OPENROUTER_URL = "https://openrouter.ai/api/v1/rerank"


def local_ranks(onnx_dir: str, items: list) -> list:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from flashindorank import CustomReranker
    from flashrank import RerankRequest

    ranker = CustomReranker(onnx_dir)
    ranks = []
    for it in items:
        passages = [{"id": "POS", "text": it["positive"]}] + [
            {"id": i, "text": neg} for i, neg in enumerate(it["negatives"])
        ]
        out = ranker.rerank(RerankRequest(query=it["query"], passages=passages))
        order = [p["id"] for p in out]
        ranks.append(order.index("POS") + 1)
    return ranks


def openrouter_ranks(model: str, items: list, key: str, pause: float = 0.3) -> list:
    ranks = []
    for it in items:
        documents = [it["positive"]] + list(it["negatives"])  # index 0 == positive
        body = json.dumps({"model": model, "query": it["query"], "documents": documents}).encode()
        req = urllib.request.Request(
            OPENROUTER_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/madebyaris/flashIndorank",
                "X-Title": "flashIndorank",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.load(resp)
            results = sorted(data["results"], key=lambda r: r["relevance_score"], reverse=True)
            order = [r["index"] for r in results]
            ranks.append(order.index(0) + 1)
        except urllib.error.HTTPError as e:
            print(f"  ! {model} HTTP {e.code}: {e.read().decode()[:160]}", file=sys.stderr)
            ranks.append(0)  # treat as miss
        except Exception as e:  # noqa: BLE001
            print(f"  ! {model} error: {repr(e)[:160]}", file=sys.stderr)
            ranks.append(0)
        time.sleep(pause)
    return ranks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", default="data/eval.jsonl")
    parser.add_argument("--n", type=int, default=15)
    parser.add_argument("--onnx-dir", default="models/ft-id-ce-onnx")
    parser.add_argument(
        "--openrouter-models",
        nargs="+",
        default=["cohere/rerank-v3.5", "nvidia/llama-nemotron-rerank-vl-1b-v2:free"],
    )
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("Set OPENROUTER_API_KEY in the environment.")

    items = read_jsonl(args.eval_file)[: args.n]
    n_docs = 1 + len(items[0]["negatives"])
    print(f"Comparing on {len(items)} Indonesian queries (1 positive + {n_docs - 1} negatives each)\n")

    rows = []
    print("Scoring local fine-tuned ONNX model ...", file=sys.stderr)
    rows.append(("flashindorank ft-id-ce (ONNX, ours)", aggregate(local_ranks(args.onnx_dir, items), k=args.k)))
    for model in args.openrouter_models:
        print(f"Scoring {model} via OpenRouter ...", file=sys.stderr)
        rows.append((model, aggregate(openrouter_ranks(model, items, key), k=args.k)))

    header = f"{'model':<46}{'top1':>8}{'MRR':>8}{f'nDCG@{args.k}':>10}"
    print(header)
    print("-" * len(header))
    for name, m in rows:
        print(f"{name:<46}{m['top1_acc']:>8.3f}{m['mrr']:>8.3f}{m[f'ndcg@{args.k}']:>10.3f}")


if __name__ == "__main__":
    main()
