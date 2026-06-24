"""Hard-negative mining for Indonesian cross-encoder training.

Combines lexical overlap, BM25, and optional dense retrieval — the pattern used
by Japanese / BGE reranker builders (BM25 + embedding models).
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Set

from common import LexicalIndex, tokenize
from miracl_data import MiraclBM25


class HardNegativeMiner:
    """Mine hard negatives for a query given a passage pool and optional BM25."""

    def __init__(
        self,
        passages: Sequence[str],
        passage_ids: Sequence[str] | None = None,
        bm25: MiraclBM25 | None = None,
        dense_model_name: str | None = None,
        seed: int = 42,
    ):
        self.passages = list(passages)
        self.passage_ids = list(passage_ids) if passage_ids else [str(i) for i in range(len(passages))]
        self.id_to_idx = {pid: i for i, pid in enumerate(self.passage_ids)}
        self.idx_to_passage = {i: p for i, p in enumerate(self.passages)}
        # Built lazily: a LexicalIndex over a large pool (e.g. the 1.44M MIRACL
        # corpus) allocates one Counter per passage. When lexical mining is
        # disabled (lexical_frac=0) it is never queried, so building it eagerly
        # wastes ~10-15 GB of RAM and re-tokenizes the corpus a second time.
        self._lexical: LexicalIndex | None = None
        self.bm25 = bm25
        self._dense = None
        self._dense_name = dense_model_name
        self._passage_emb = None
        random.seed(seed)

    @property
    def lexical(self) -> LexicalIndex:
        if self._lexical is None:
            self._lexical = LexicalIndex(self.passages)
        return self._lexical

    def _dense_model(self):
        if self._dense is None and self._dense_name:
            import torch
            from sentence_transformers import SentenceTransformer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            encode_batch = 256 if device == "cuda" else 64
            print(f"Loading dense retriever {self._dense_name} on {device} ...", flush=True)
            self._dense = SentenceTransformer(self._dense_name, device=device)
            print(f"Encoding {len(self.passages)} passages for dense mining ...", flush=True)
            self._passage_emb = self._dense.encode(
                self.passages,
                normalize_embeddings=True,
                show_progress_bar=True,
                batch_size=encode_batch,
            )
            print("Dense passage index ready.", flush=True)
        return self._dense

    def _dense_top(self, query: str, k: int, exclude_idx: int) -> List[int]:
        if not self._dense_name:
            return []
        self._dense_model()
        import numpy as np

        q_emb = self._dense.encode([query], normalize_embeddings=True)[0]
        scores = self._passage_emb @ q_emb
        ranked = np.argsort(-scores)
        out: List[int] = []
        for idx in ranked:
            i = int(idx)
            if i == exclude_idx:
                continue
            out.append(i)
            if len(out) >= k:
                break
        return out

    def mine(
        self,
        query: str,
        positive_id: str | int,
        n: int,
        *,
        exclude_ids: Set[str] | None = None,
        bm25_k: int = 200,
        lexical_frac: float = 0.2,
        dense_frac: float = 0.3,
    ) -> List[str]:
        """Return ``n`` hard-negative passage texts for ``query``."""
        exclude_ids = set(exclude_ids or [])
        if isinstance(positive_id, int):
            gold_idx = positive_id
            exclude_ids.add(self.passage_ids[gold_idx])
        else:
            gold_idx = self.id_to_idx.get(str(positive_id), -1)
            exclude_ids.add(str(positive_id))

        n_lex = max(0, int(round(n * lexical_frac)))
        n_dense = max(0, int(round(n * dense_frac))) if self._dense_name else 0
        n_bm25 = n - n_lex - n_dense

        candidates: List[str] = []
        seen: Set[str] = set(exclude_ids)

        if gold_idx >= 0:
            seen.add(self.passage_ids[gold_idx])

        if self.bm25 is not None and n_bm25 > 0:
            for docid, text, _score in self.bm25.search(query, k=bm25_k, exclude=seen):
                if text not in candidates:
                    candidates.append(text)
                    seen.add(docid)
                if len(candidates) >= n_bm25 * 3:
                    break

        if n_lex > 0 and gold_idx >= 0:
            for idx in self.lexical.top_overlap(query, k=n_lex * 5, exclude_idx=gold_idx):
                pid = self.passage_ids[idx]
                if pid not in seen:
                    candidates.append(self.passages[idx])
                    seen.add(pid)

        if n_dense > 0 and gold_idx >= 0:
            for idx in self._dense_top(query, k=n_dense * 5, exclude_idx=gold_idx):
                pid = self.passage_ids[idx]
                if pid not in seen:
                    candidates.append(self.passages[idx])
                    seen.add(pid)

        # Top up with random pool negatives if mining came up short.
        while len(candidates) < n:
            j = random.randrange(len(self.passages))
            pid = self.passage_ids[j]
            if pid in seen:
                continue
            candidates.append(self.passages[j])
            seen.add(pid)

        random.shuffle(candidates)
        return candidates[:n]
