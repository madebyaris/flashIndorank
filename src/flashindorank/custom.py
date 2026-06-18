"""Serve a locally fine-tuned / exported ONNX cross-encoder.

FlashRank only knows about its own bundled models, so this provides a parallel
path for *custom* ONNX sequence-classification rerankers (e.g. the Indonesian
model produced by the training pipeline). It uses the fast ``tokenizers``
library directly (already a FlashRank dependency), so serving a custom model
needs no extra runtime dependencies (no torch / transformers).

The public surface intentionally mirrors ``flashrank.Ranker``: a ``rerank``
method that takes a ``RerankRequest`` and returns passages sorted by score.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


class CustomReranker:
    """ONNX cross-encoder reranker backed by a local model directory.

    The directory must contain ``model.onnx`` and a ``tokenizer.json``
    (produced by ``training/export_onnx.py``).
    """

    def __init__(
        self,
        model_dir: str,
        max_length: int = 512,
        intra_op_threads: int = 0,
        inter_op_threads: int = 0,
    ):
        self.model_dir = Path(model_dir)
        onnx_path = self.model_dir / "model.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(f"model.onnx not found in {self.model_dir}")

        so = ort.SessionOptions()
        if intra_op_threads > 0:
            so.intra_op_num_threads = intra_op_threads
        if inter_op_threads > 0:
            so.inter_op_num_threads = inter_op_threads

        self.session = ort.InferenceSession(str(onnx_path), sess_options=so)
        self._input_names = {i.name for i in self.session.get_inputs()}
        self.tokenizer = self._load_tokenizer(max_length)

    def _load_tokenizer(self, max_length: int) -> Tokenizer:
        tok_path = self.model_dir / "tokenizer.json"
        if not tok_path.exists():
            raise FileNotFoundError(f"tokenizer.json not found in {self.model_dir}")
        tokenizer = Tokenizer.from_file(str(tok_path))

        model_max = max_length
        cfg = self.model_dir / "tokenizer_config.json"
        if cfg.exists():
            with open(cfg, encoding="utf-8") as f:
                tok_cfg = json.load(f)
            mm = tok_cfg.get("model_max_length")
            if isinstance(mm, int) and mm > 0:
                model_max = min(mm, max_length)

        tokenizer.enable_truncation(max_length=model_max)
        tokenizer.enable_padding()
        return tokenizer

    def rerank(self, request: Any) -> List[Dict[str, Any]]:
        passages = request.passages
        if not passages:
            return []
        query = request.query

        pairs = [(query, p["text"]) for p in passages]
        encoded = self.tokenizer.encode_batch(pairs)
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

        onnx_input: Dict[str, np.ndarray] = {}
        if "input_ids" in self._input_names:
            onnx_input["input_ids"] = input_ids
        if "attention_mask" in self._input_names:
            onnx_input["attention_mask"] = attention_mask
        if "token_type_ids" in self._input_names:
            token_type_ids = np.array([e.type_ids for e in encoded], dtype=np.int64)
            onnx_input["token_type_ids"] = token_type_ids

        logits = self.session.run(None, onnx_input)[0]
        if logits.shape[1] == 1:
            scores = 1.0 / (1.0 + np.exp(-logits.flatten()))
        else:
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            scores = exp[:, 1] / np.sum(exp, axis=1)

        for score, passage in zip(scores, passages):
            passage["score"] = float(score)
        passages.sort(key=lambda x: x["score"], reverse=True)
        return passages
