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
import threading
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

        # Default to single-threaded ORT: multi-threaded intra-op execution is
        # not deterministic when driven from a threadpool (it can even reorder
        # results), and on a cheap VPS the per-request lock serializes work
        # anyway. Callers can still opt into more threads explicitly.
        so = ort.SessionOptions()
        so.intra_op_num_threads = intra_op_threads if intra_op_threads > 0 else 1
        so.inter_op_num_threads = inter_op_threads if inter_op_threads > 0 else 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # The shared CPU memory arena is process-global; disabling it keeps
        # results identical regardless of which (threadpool) thread runs.
        so.enable_cpu_mem_arena = False

        self.session = ort.InferenceSession(str(onnx_path), sess_options=so)
        self._input_names = {i.name for i in self.session.get_inputs()}
        self.tokenizer = self._load_tokenizer(max_length)
        # The fast tokenizer carries mutable padding/truncation state and is not
        # safe to drive from multiple threads at once (e.g. under uvicorn's
        # threadpool). Serialize tokenize + inference to keep results correct.
        self._lock = threading.Lock()

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

        # Use the model's real pad token id when we can find it.
        pad_id, pad_token = 0, "[PAD]"
        config_path = self.model_dir / "config.json"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                model_cfg = json.load(f)
            if isinstance(model_cfg.get("pad_token_id"), int):
                pad_id = model_cfg["pad_token_id"]
                pad_token = tokenizer.id_to_token(pad_id) or pad_token

        tokenizer.enable_truncation(max_length=model_max)
        # Padding is done manually in numpy (see ``rerank``); the batched
        # tokenizer padding path uses internal parallelism that can race and
        # produce inconsistent encodings, so we keep the tokenizer single-shot.
        tokenizer.no_padding()
        self._pad_id = pad_id
        return tokenizer

    def rerank(self, request: Any) -> List[Dict[str, Any]]:
        passages = request.passages
        if not passages:
            return []
        query = request.query

        with self._lock:
            # Encode each (query, passage) pair individually (no batch-level
            # parallelism) then pad to the batch max length ourselves.
            encoded = [self.tokenizer.encode(query, p["text"]) for p in passages]
            max_len = max(len(e.ids) for e in encoded)
            n = len(encoded)

            input_ids = np.full((n, max_len), self._pad_id, dtype=np.int64)
            attention_mask = np.zeros((n, max_len), dtype=np.int64)
            token_type_ids = np.zeros((n, max_len), dtype=np.int64)
            for i, e in enumerate(encoded):
                length = len(e.ids)
                input_ids[i, :length] = e.ids
                attention_mask[i, :length] = e.attention_mask
                token_type_ids[i, :length] = e.type_ids

            onnx_input: Dict[str, np.ndarray] = {}
            if "input_ids" in self._input_names:
                onnx_input["input_ids"] = input_ids
            if "attention_mask" in self._input_names:
                onnx_input["attention_mask"] = attention_mask
            if "token_type_ids" in self._input_names:
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
