# AGENTS.md

## Cursor Cloud specific instructions

**Product:** `flashindorank` — a lightweight, high-performance cross-encoder
reranker built on [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank),
tuned to run fast on cheap VPS hardware. Python 3.12, packaged with a `src/`
layout (`src/flashindorank`). The HTTP service is FastAPI + uvicorn.

There is a single service (the FastAPI reranker API). No database or external
services are required.

### Layout
- `src/flashindorank/` — package: `config.py` (env settings), `models.py`
  (model registry), `reranker.py` (core + cascade), `api.py` (FastAPI app),
  `__main__.py` (server entry).
- `benchmarks/benchmark.py` — latency/throughput benchmark.
- `tests/` — pytest suite (`test_reranker.py`, `test_api.py`).

### Standard commands (see README for full details)
- Lint: `ruff check src benchmarks tests`
- Test: `pytest`
- Run API (dev): `python -m flashindorank` (serves on `:8000`, docs at `/docs`)
- Benchmark: `python benchmarks/benchmark.py`

### Non-obvious gotchas
- **Use a virtualenv.** The system Python is externally managed (PEP 668); the
  update script creates/uses `/workspace/.venv`. Activate it with
  `source .venv/bin/activate` before running any command. `python3-venv` is a
  system package that must already be present (it is on the base VM).
- **Model weights download on first use** from Hugging Face
  (`huggingface.co/prithivida/flashrank`); the first rerank/test needs network.
  Set `FLASHINDORANK_CACHE_DIR` to a persistent dir to avoid re-downloading
  (tests/`conftest.py` point it at `/workspace/.model_cache`). Models are tiny
  (TinyBERT 3.3 MB, MiniLM 21.6 MB).
- **FlashRank does not expose ONNX SessionOptions**, so `reranker.py` rebuilds
  the session to apply `FLASHINDORANK_INTRA_OP_THREADS` /
  `FLASHINDORANK_INTER_OP_THREADS`. Pin these to 1–2 on small VPS boxes.
- Rankers are cached as warm singletons keyed by `(model, max_length)`; the
  first request per model pays the load cost, set `FLASHINDORANK_PRELOAD_MODELS`
  to warm them at startup.

### Indonesian language note (project focus)
The bundled default models are English MS-MARCO cross-encoders and are **weak on
Indonesian semantics** (they rely on lexical overlap). `benchmarks/eval_indonesian.py`
is a quick reproducible harness for that claim.

The recommended path for strong Indonesian relevance is the **fine-tuning
pipeline** in `training/` (documented in `TRAINING.md`): prepare data from TyDi
QA → fine-tune a multilingual cross-encoder → export to quantized ONNX → serve
via the `CustomReranker` using `FLASHINDORANK_CUSTOM_MODELS=name=/path`. On a
reference CPU run the fine-tuned model reached top-1 0.895 / MRR 0.940 vs 0.615 /
0.743 for the English default.

Gotchas for the training pipeline:
- Training deps are **heavy and separate**: `pip install -r requirements-train.txt`
  (CPU torch). They are NOT part of the startup update script.
- `datasets` 5.x dropped script-based loaders, so script datasets like
  `unicamp-dl/mmarco` / `miracl` won't `load_dataset` directly; we use TyDi QA
  (`tydiqa`, `secondary_task`), filtering Indonesian via the `id` field prefix.
- `training/` scripts import sibling modules (`from common import ...`), so run
  them from the repo root (e.g. `python training/train.py`).
- Custom ONNX serving needs only the runtime deps (uses the `tokenizers` lib, no
  torch/transformers at inference). The exported dir must contain `model.onnx`
  + `tokenizer.json`.
- `CustomReranker` intentionally pins ORT to a single thread, disables the CPU
  mem arena, pads manually (not via `encode_batch`), and serializes inference
  with a lock. This is required for deterministic, correct rankings when served
  under uvicorn's threadpool; multi-threaded ORT was observed to reorder results.
- Trained/exported models and datasets live under `models/` and `data/` and are
  gitignored (do not commit weights).

### Available runtimes on the VM
Python 3.12, `pip`, Node.js 22 (`npm`, `pnpm`), Go 1.22 (`uv` not installed).
This project only uses Python.
