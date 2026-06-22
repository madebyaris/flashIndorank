# Fine-tuning an Indonesian reranker

This pipeline fine-tunes a lightweight multilingual cross-encoder into a
stronger **Indonesian** reranker, exports it to a quantized ONNX model, and
serves it through flashIndorank — all CPU-friendly so it still fits a cheap VPS.

## TL;DR

```bash
# 1. Install training deps (heavy; separate from runtime deps)
pip install -r requirements-train.txt

# 2. Build Indonesian train/eval data (TyDi QA + MIRACL-id, BM25 hard negatives)
python training/prepare_data.py --sources tydi,miracl --train-negatives 15

# 3. Fine-tune v2 (1 epoch default; GPU recommended for full data)
python training/train.py --epochs 1 --batch-size 16 --out-dir models/ft-id-ce-v2

# 4. Compare on TyDi holdout + MIRACL-id official rerank eval
python training/evaluate.py --models \
    cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 \
    models/ft-id-ce-v2
python training/eval_miracl.py --model models/ft-id-ce-v2

# 5. Export to quantized ONNX for lightweight serving
python training/export_onnx.py --model-dir models/ft-id-ce-v2 --out-dir models/ft-id-ce-v2-onnx

# 6. Serve the fine-tuned model through the API
export FLASHINDORANK_CUSTOM_MODELS="id-reranker=$PWD/models/ft-id-ce-onnx"
python -m flashindorank
# -> POST /rerank with {"model": "id-reranker", ...}

# 7. (Optional) Publish to the Hugging Face Hub
export HF_TOKEN=hf_xxx   # a token with Write access
python training/upload_to_hf.py --repo-id madebyaris/rerank-indonesia
```

## Publishing to Hugging Face

`training/upload_to_hf.py` pushes both formats to one repo:

- the sentence-transformers CrossEncoder (PyTorch/safetensors) at the repo root,
  so `CrossEncoder("madebyaris/rerank-indonesia")` works, and
- the quantized ONNX model + tokenizer under `onnx/` for lightweight serving,

plus a generated model card. It needs a **write** token in `HF_TOKEN` (or
`HUGGING_FACE_HUB_TOKEN`); create one at
<https://huggingface.co/settings/tokens>.

## How it works

| Step | Script | What it does |
| --- | --- | --- |
| Data | `training/prepare_data.py` | Loads **TyDi QA** Indonesian rows and **MIRACL-id train** (topics+qrels+corpus). Mines **BM25 hard negatives** (15/query default) plus optional dense mining on the TyDi passage pool. Writes `data/train.jsonl` and `data/eval.jsonl`. |
| Train | `training/train.py` | Fine-tunes a `CrossEncoder` (default base `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`) with binary cross-entropy. Default output: `models/ft-id-ce-v2`. |
| Eval | `training/evaluate.py` | Reports top-1 accuracy, MRR, nDCG@10 on the TyDi holdout set. |
| MIRACL eval | `training/eval_miracl.py` | Official-style protocol: BM25 top-100 from the MIRACL-id corpus → cross-encoder rerank → nDCG@10 / MRR@10 / Recall@100. |
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
