"""Build Indonesian cross-encoder train/eval data (v2).

Sources:
* **TyDi QA** Indonesian Gold Passage rows (native human QA)
* **MIRACL-id train** topics + qrels over the Wikipedia corpus

Hard negatives are mined with BM25 over the MIRACL corpus (and optional dense
retrieval over smaller pools), following the Japanese / BGE reranker recipe.

Outputs (under --out-dir):
* train.jsonl  - {"query", "passage", "label"} pairs
* eval.jsonl   - TyDi holdout ranking items for the legacy evaluate.py harness

Run (venv with training deps active):

    python training/prepare_data.py --sources tydi,miracl --train-negatives 15
"""

from __future__ import annotations

import argparse
import random
import warnings
from pathlib import Path

from common import LexicalIndex, write_jsonl
from hard_negatives import HardNegativeMiner
from miracl_data import MiraclBM25, load_qrels, load_topics, qrels_path, topics_path

warnings.filterwarnings("ignore")


def _load_indonesian_tydi_rows():
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
            rows.append({"query": q, "passage": ctx, "source": "tydi"})
    return rows, list(seen_passages.keys())


def _load_miracl_train_rows(bm25: MiraclBM25, cache: str) -> list[dict]:
    topics = load_topics(topics_path("train", cache))
    qrels = load_qrels(qrels_path("train", cache))
    rows = []
    for qid, rel_docs in qrels.items():
        query = topics.get(qid)
        if not query or not rel_docs:
            continue
        # One positive per query (first relevant docid) — matches the 1pos+15neg recipe.
        docid = next(iter(rel_docs))
        idx = bm25.docid_to_idx.get(docid)
        if idx is None:
            continue
        rows.append(
            {
                "query": query,
                "passage": bm25.texts[idx],
                "positive_id": docid,
                "source": "miracl",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--miracl-cache", default="data/miracl")
    parser.add_argument(
        "--sources",
        default="tydi,miracl",
        help="comma-separated: tydi, miracl",
    )
    parser.add_argument("--train-negatives", type=int, default=15)
    parser.add_argument("--eval-negatives", type=int, default=9)
    parser.add_argument("--tydi-eval-queries", type=int, default=200)
    parser.add_argument("--max-tydi-train", type=int, default=0, help="0 = all TyDi train rows")
    parser.add_argument("--max-miracl-train", type=int, default=0, help="0 = all MIRACL train qids")
    parser.add_argument("--max-corpus-docs", type=int, default=0, help="0 = full MIRACL corpus")
    parser.add_argument(
        "--dense-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="dense retriever for TyDi-pool mining (empty to disable)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    sources = {s.strip() for s in args.sources.split(",") if s.strip()}

    tydi_rows: list[dict] = []
    tydi_passages: list[str] = []
    if "tydi" in sources:
        print("Loading Indonesian TyDi QA rows...")
        tydi_rows, tydi_passages = _load_indonesian_tydi_rows()
        print(f"  {len(tydi_rows)} rows, {len(tydi_passages)} unique passages")

    bm25: MiraclBM25 | None = None
    if "miracl" in sources:
        print("Loading MIRACL-id corpus + building BM25 index (one-time, ~1.4M passages)...")
        bm25 = MiraclBM25.from_cache(cache=args.miracl_cache, max_docs=args.max_corpus_docs)
        print(f"  indexed {len(bm25.docids)} passages")

    miracl_rows: list[dict] = []
    if "miracl" in sources and bm25 is not None:
        print("Building MIRACL-id train rows from topics + qrels...")
        miracl_rows = _load_miracl_train_rows(bm25, args.miracl_cache)
        print(f"  {len(miracl_rows)} (query, positive passage) rows")

    # ---- TyDi train / eval split ----
    random.shuffle(tydi_rows)
    eval_rows = tydi_rows[: args.tydi_eval_queries] if tydi_rows else []
    train_tydi = tydi_rows[args.tydi_eval_queries :]
    if args.max_tydi_train and len(train_tydi) > args.max_tydi_train:
        train_tydi = train_tydi[: args.max_tydi_train]

    # ---- MIRACL train cap ----
    if args.max_miracl_train and len(miracl_rows) > args.max_miracl_train:
        random.shuffle(miracl_rows)
        miracl_rows = miracl_rows[: args.max_miracl_train]

    # TyDi miner uses the smaller passage pool (+ optional dense); MIRACL uses corpus BM25.
    tydi_miner = None
    if train_tydi:
        passage_to_idx = {p: i for i, p in enumerate(tydi_passages)}
        tydi_miner = HardNegativeMiner(
            tydi_passages,
            passage_ids=[str(i) for i in range(len(tydi_passages))],
            bm25=None,
            dense_model_name=args.dense_model or None,
            seed=args.seed,
        )
        # Re-use lexical index from miner for eval negatives on TyDi pool.
        tydi_lexical = tydi_miner.lexical
    else:
        passage_to_idx = {}
        tydi_lexical = None

    def tydi_sample_negs(query: str, gold_idx: int, n: int) -> list[str]:
        if tydi_miner is not None:
            return tydi_miner.mine(query, gold_idx, n, bm25_k=0, lexical_frac=0.4, dense_frac=0.6)
        return []

    train_pairs: list[dict] = []

    print(f"Mining negatives for {len(train_tydi)} TyDi train queries...")
    for i, r in enumerate(train_tydi):
        gold_idx = passage_to_idx[r["passage"]]
        train_pairs.append({"query": r["query"], "passage": r["passage"], "label": 1.0})
        for neg in tydi_sample_negs(r["query"], gold_idx, args.train_negatives):
            train_pairs.append({"query": r["query"], "passage": neg, "label": 0.0})
        if (i + 1) % 500 == 0:
            print(f"  TyDi {i + 1}/{len(train_tydi)}")

    if miracl_rows and bm25 is not None:
        miracl_miner = HardNegativeMiner(
            bm25.texts,
            passage_ids=bm25.docids,
            bm25=bm25,
            dense_model_name=None,
            seed=args.seed,
        )
        print(f"Mining negatives for {len(miracl_rows)} MIRACL train rows...")
        for i, r in enumerate(miracl_rows):
            train_pairs.append({"query": r["query"], "passage": r["passage"], "label": 1.0})
            negs = miracl_miner.mine(
                r["query"],
                r["positive_id"],
                args.train_negatives,
                exclude_ids={r["positive_id"]},
                bm25_k=200,
                lexical_frac=0.0,
                dense_frac=0.0,
            )
            for neg in negs:
                train_pairs.append({"query": r["query"], "passage": neg, "label": 0.0})
            if (i + 1) % 500 == 0:
                print(f"  MIRACL {i + 1}/{len(miracl_rows)}")

    random.shuffle(train_pairs)
    n_train = write_jsonl(Path(args.out_dir) / "train.jsonl", train_pairs)

    eval_items = []
    if eval_rows and tydi_lexical is not None:
        for r in eval_rows:
            gold_idx = passage_to_idx[r["passage"]]
            n_hard = int(round(args.eval_negatives * 0.7))
            hard = [
                tydi_passages[j]
                for j in tydi_lexical.top_overlap(r["query"], k=n_hard * 3, exclude_idx=gold_idx)
            ][:n_hard]
            negs = list(hard)
            while len(negs) < args.eval_negatives:
                j = random.randrange(len(tydi_passages))
                if j != gold_idx and tydi_passages[j] not in negs:
                    negs.append(tydi_passages[j])
            eval_items.append(
                {"query": r["query"], "positive": r["passage"], "negatives": negs[: args.eval_negatives]}
            )
    n_eval = write_jsonl(Path(args.out_dir) / "eval.jsonl", eval_items)

    print(f"Wrote {n_train} train pairs -> {args.out_dir}/train.jsonl")
    print(f"Wrote {n_eval} eval items  -> {args.out_dir}/eval.jsonl")
    print(
        f"  sources: {sorted(sources)} | negatives/query: {args.train_negatives} | "
        f"TyDi train: {len(train_tydi)} | MIRACL train: {len(miracl_rows)}"
    )


if __name__ == "__main__":
    main()
