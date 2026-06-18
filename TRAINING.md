# Fine-tuning an Indonesian reranker

This pipeline fine-tunes a lightweight multilingual cross-encoder into a
stronger **Indonesian** reranker, exports it to a quantized ONNX model, and
serves it through flashIndorank — all CPU-friendly so it still fits a cheap VPS.

## TL;DR

```bash
# 1. Install training deps (heavy; separate from runtime deps)
pip install -r requirements-train.txt

# 2. Build Indonesian train/eval data from TyDi QA (Gold Passage)
python training/prepare_data.py --train-queries 800 --eval-queries 200

# 3. Fine-tune (CPU smoke run ~10 min; use a GPU + more data for real runs)
python training/train.py --epochs 2 --batch-size 16

# 4. Compare base vs fine-tuned (and the English default) on the SAME eval
python training/evaluate.py --models \
    ms-marco-MiniLM-L-12-v2 \
    cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 \
    models/ft-id-ce

# 5. Export to quantized ONNX for lightweight serving
python training/export_onnx.py --model-dir models/ft-id-ce --out-dir models/ft-id-ce-onnx

# 6. Serve the fine-tuned model through the API
export FLASHINDORANK_CUSTOM_MODELS="id-reranker=$PWD/models/ft-id-ce-onnx"
python -m flashindorank
# -> POST /rerank with {"model": "id-reranker", ...}
```

## How it works

| Step | Script | What it does |
| --- | --- | --- |
| Data | `training/prepare_data.py` | Loads Indonesian rows from **TyDi QA** (`secondary_task`/Gold Passage), takes each gold passage as a positive, and mines **hard negatives** by lexical overlap. Writes `data/train.jsonl` (pairs + labels) and `data/eval.jsonl` (ranking items). |
| Train | `training/train.py` | Fine-tunes a `CrossEncoder` (default base `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`) with binary cross-entropy using sentence-transformers v5 `CrossEncoderTrainer`. |
| Eval | `training/evaluate.py` | Reports top-1 accuracy, MRR, nDCG@10 for any mix of FlashRank bundled models and CrossEncoder paths/names. |
| Export | `training/export_onnx.py` | Exports to ONNX via `optimum` and applies dynamic int8 quantization, producing `model.onnx` + tokenizer. |
| Serve | `flashindorank` `CustomReranker` | Loads the local ONNX model with the fast `tokenizers` lib (no torch/transformers at inference) and serves it like any bundled model. |

## Results from the reference CPU run

Held-out Indonesian eval, 200 queries, 1 positive + 9 hard negatives each:

| model | top-1 | MRR | nDCG@10 |
| --- | --- | --- | --- |
| `ms-marco-MiniLM-L-12-v2` (English default) | 0.615 | 0.743 | 0.805 |
| `ms-marco-MultiBERT-L-12` (FlashRank multilingual) | 0.390 | 0.561 | 0.664 |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (base) | 0.860 | 0.921 | 0.941 |
| **`models/ft-id-ce` (our fine-tune)** | **0.895** | **0.940** | **0.956** |

Fine-tuning on Indonesian beat the strong multilingual base on every metric and
massively outperformed the English default — confirming the dataset+pipeline
works. For production, use more data (full TyDi QA / MIRACL-id / mMARCO-id),
more epochs, and a GPU; everything above is parameterized via CLI flags.

## Notes / knobs

- Base model: `--base-model` (e.g. continue from a different multilingual CE).
- Data scale: `--train-queries`, `--train-negatives`, `--hard-frac`.
- Training: `--epochs`, `--batch-size`, `--lr`, `--max-length`, `--max-train`.
- Export: `--no-quantize` to keep fp32.
- Add your own domain queries by extending `data/*.jsonl` (same JSON schema).
