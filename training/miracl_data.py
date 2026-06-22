"""MIRACL Indonesian data helpers (TSV + corpus shards).

``datasets`` 5.x no longer runs the MIRACL loading script, so we download the
published TSV / JSONL.gz files directly from the Hugging Face dataset repo and
build a BM25 index locally for retrieval-style eval and hard-negative mining.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from huggingface_hub import hf_hub_download
from rank_bm25 import BM25Okapi

from common import tokenize

REPO_ID = "miracl/miracl"
CORPUS_REPO = "miracl/miracl-corpus"
LANG = "id"


def _cache_dir(cache: str | Path) -> Path:
    path = Path(cache)
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_file(repo: str, filename: str, cache: str | Path) -> Path:
    """Download a dataset file from the Hub if it is not already cached."""
    dest = _cache_dir(cache) / filename.replace("/", "__")
    if dest.exists():
        return dest
    downloaded = Path(hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset"))
    import shutil

    shutil.copy2(downloaded, dest)
    return dest


def topics_path(split: str, cache: str | Path = "data/miracl") -> Path:
    return download_file(REPO_ID, f"miracl-v1.0-{LANG}/topics/topics.miracl-v1.0-{LANG}-{split}.tsv", cache)


def qrels_path(split: str, cache: str | Path = "data/miracl") -> Path:
    return download_file(REPO_ID, f"miracl-v1.0-{LANG}/qrels/qrels.miracl-v1.0-{LANG}-{split}.tsv", cache)


def load_topics(path: str | Path) -> Dict[str, str]:
    """Return ``{query_id: query_text}`` from a MIRACL topics TSV."""
    out: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            qid, query = line.split("\t", 1)
            out[qid] = query.strip()
    return out


def load_qrels(path: str | Path) -> Dict[str, Dict[str, int]]:
    """Return ``{query_id: {docid: relevance}}`` (TREC qrels format)."""
    out: Dict[str, Dict[str, int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, _iter, docid, rel = parts[0], parts[1], parts[2], int(parts[3])
            if rel > 0:
                out.setdefault(qid, {})[docid] = rel
    return out


def corpus_shard_paths(cache: str | Path = "data/miracl") -> List[Path]:
    shards = []
    for i in range(3):
        name = f"miracl-corpus-v1.0-{LANG}/docs-{i}.jsonl.gz"
        shards.append(download_file(CORPUS_REPO, name, cache))
    return shards


def iter_corpus(
    cache: str | Path = "data/miracl",
    max_docs: int = 0,
) -> Iterable[Tuple[str, str]]:
    """Yield ``(docid, passage_text)`` from gzipped JSONL corpus shards."""
    n = 0
    for shard in corpus_shard_paths(cache):
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                docid = str(row["docid"])
                title = (row.get("title") or "").strip()
                text = (row.get("text") or "").strip()
                passage = f"{title}. {text}".strip(". ").strip() if title else text
                if passage:
                    yield docid, passage
                    n += 1
                    if max_docs and n >= max_docs:
                        return


def load_corpus(
    cache: str | Path = "data/miracl",
    max_docs: int = 0,
) -> Tuple[List[str], List[str]]:
    """Load corpus into parallel ``docids`` and ``texts`` lists."""
    docids: List[str] = []
    texts: List[str] = []
    for i, (docid, text) in enumerate(iter_corpus(cache=cache, max_docs=max_docs), start=1):
        docids.append(docid)
        texts.append(text)
        if i % 200_000 == 0:
            print(f"  loaded {i} corpus passages...", flush=True)
    print(f"  loaded {len(docids)} corpus passages total.", flush=True)
    return docids, texts


class MiraclBM25:
    """BM25 index over the MIRACL Indonesian Wikipedia corpus."""

    def __init__(self, docids: Sequence[str], texts: Sequence[str]):
        self.docids = list(docids)
        self.texts = list(texts)
        self.docid_to_idx = {d: i for i, d in enumerate(self.docids)}
        print(f"Tokenizing {len(self.texts)} passages for BM25 ...", flush=True)
        self._tokenized = [tokenize(t) for t in self.texts]
        print("Building BM25 index ...", flush=True)
        self._bm25 = BM25Okapi(self._tokenized)
        print("BM25 index ready.", flush=True)

    def search(self, query: str, k: int, exclude: set[str] | None = None) -> List[Tuple[str, str, float]]:
        """Return up to ``k`` ``(docid, text, score)`` hits excluding ``exclude`` docids."""
        exclude = exclude or set()
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits: List[Tuple[str, str, float]] = []
        for idx in ranked:
            if scores[idx] <= 0:
                break
            docid = self.docids[idx]
            if docid in exclude:
                continue
            hits.append((docid, self.texts[idx], float(scores[idx])))
            if len(hits) >= k:
                break
        return hits

    @classmethod
    def from_cache(cls, cache: str | Path = "data/miracl", max_docs: int = 0) -> "MiraclBM25":
        docids, texts = load_corpus(cache=cache, max_docs=max_docs)
        return cls(docids, texts)
