# Training an Indonesian reranker (knowledge distillation)

This is how the published **[`madebyaris/rerank-indonesia`](https://huggingface.co/madebyaris/rerank-indonesia)**
model is built. We distill a strong but heavy teacher into a tiny, VPS-friendly
student, then export a quantized ONNX model — all parameterized via CLI flags.

The winning recipe is **Margin-MSE knowledge distillation** on **in-domain**
Indonesian data: a 568M teacher (`BAAI/bge-reranker-v2-m3`) scores
(query, positive) and (query, negative) pairs, and the tiny student learns the
teacher's *margin* `teacher(q,pos) − teacher(q,neg)`. The teacher scoring and
training need a **GPU** (we use a RunPod RTX 3090); export/serving are CPU-only.

## TL;DR

```bash
# 0. Install training deps (heavy; separate from runtime deps)
pip install -r requirements-train.txt

# --- Option A: in-domain distillation (what we shipped, best on MIRACL) ---
# 1. Build in-domain pairs (TyDi QA + MIRACL-id, BM25/dense hard negatives)
python training/prepare_data.py --sources tydi,miracl --train-negatives 15
# 2. Regroup labeled pairs into (query, positive, negative) triplets
python training/pairs_to_triplets.py \
    --in data/train.jsonl --out data/indomain_triplets.jsonl --negs-per-query 6
# 3. Teacher-score the triplets -> Margin-MSE labels (GPU)
python training/teacher_score.py \
    --in data/indomain_triplets.jsonl --out data/indomain_scored.jsonl \
    --teacher BAAI/bge-reranker-v2-m3
# 4. Distill into the tiny student (GPU)
python training/train_distill.py \
    --train-file data/indomain_scored.jsonl --out-dir models/ft-id-ce-distill --epochs 3

# --- Option B: large-scale mMARCO-id distillation (more data, less in-domain) ---
python training/prepare_distill_data.py --mmarco-queries 80000 --negs-per-query 5
python training/teacher_score.py \
    --in data/distill_triplets.jsonl --out data/distill_train.jsonl \
    --teacher BAAI/bge-reranker-v2-m3
python training/train_distill.py \
    --train-file data/distill_train.jsonl --eval-file data/distill_eval.jsonl \
    --out-dir models/ft-id-ce-distill-mmarco --epochs 2

# 5. Evaluate on MIRACL-id — the meaningful benchmark
python training/eval_miracl.py --model models/ft-id-ce-distill

# 6. Export to quantized int8 ONNX for lightweight serving
python training/export_onnx.py \
    --model-dir models/ft-id-ce-distill --out-dir models/ft-id-ce-distill-onnx

# 7. Serve through the API
export FLASHINDORANK_CUSTOM_MODELS="id-reranker=$PWD/models/ft-id-ce-distill-onnx"
python -m flashindorank   # POST /rerank with {"model": "id-reranker", ...}

# 8. Publish to Hugging Face (main)
export HF_TOKEN=hf_xxx
python training/upload_to_hf.py \
    --st-dir models/ft-id-ce-distill --onnx-dir models/ft-id-ce-distill-onnx
```

> `--eval-file data/distill_eval.jsonl` (a held-out reranking set produced by
> `prepare_distill_data.py`) is optional; pass it to `train_distill.py` to get a
> `CrossEncoderRerankingEvaluator` readout during training. The final verdict
> always comes from `eval_miracl.py`.

## How it works

| Step | Script | What it does |
| --- | --- | --- |
| In-domain data | `training/prepare_data.py` | Loads **TyDi QA** Indonesian + **MIRACL-id train** (topics+qrels+corpus); mines **BM25 + dense hard negatives**. Writes `data/train.jsonl` (labeled pairs). |
| Pairs → triplets | `training/pairs_to_triplets.py` | Groups labeled pairs by query into `(query, positive, negative)` triplets. |
| mMARCO data (opt.) | `training/prepare_distill_data.py` | Streams **mMARCO-id** (`crystina-z/mmarco-passage`), which already ships 1 positive + ~30 hard negatives per query, into triplets — large scale, no 8.8M-corpus load. |
| Teacher scoring | `training/teacher_score.py` | Loads the teacher (`BAAI/bge-reranker-v2-m3`), scores raw logits (deduped pairs, batched on GPU), writes triplets with `label = teacher(q,pos) − teacher(q,neg)`. |
| Distill | `training/train_distill.py` | Trains the tiny student with `MarginMSELoss` (default base `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`), lr 8e-6, bf16. Optional `CrossEncoderRerankingEvaluator`. |
| MIRACL eval | `training/eval_miracl.py` | Official protocol: BM25 top-100 from the MIRACL-id corpus → rerank → nDCG@10 / MRR@10 / Recall@100. |
| Export | `training/export_onnx.py` | ONNX via `optimum` + dynamic int8 quantization → `model.onnx` + tokenizer (~118 MB). |
| Serve | `flashindorank` `CustomReranker` | Loads the local ONNX with the fast `tokenizers` lib (no torch/transformers at inference). |
| Publish | `training/upload_to_hf.py` | Pushes the sentence-transformers weights (repo root) + int8 ONNX (`onnx/`) + model card. Needs a **write** `HF_TOKEN`. |

## Results

**MIRACL-id** dev, official retrieve-then-rerank (BM25 top-100 → rerank, 960
queries, `pytrec_eval`):

| model | size | nDCG@10 | MRR@10 | Recall@100 |
| --- | --- | --- | --- | --- |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (base) | tiny | 0.656 | 0.623 | 0.760 |
| BCE fine-tune (in-domain) | tiny | 0.692 | 0.669 | 0.760 |
| distillation on mMARCO-id (400k triplets) | tiny | 0.687 | 0.660 | 0.760 |
| **distillation on in-domain (shipped)** | **tiny** | **0.701** | **0.677** | 0.760 |
| `BAAI/bge-reranker-v2-m3` (teacher) | 568M | 0.712 | 0.689 | 0.760 |

The shipped distilled student beats the base by **+4.5 nDCG / +5.4 MRR** and
lands within ~1.1 nDCG of the 17× larger teacher (~98% of its quality). For
MIRACL, in-domain data beat raw mMARCO scale. `Recall@100 = 0.760` is the BM25
first-stage ceiling and caps every reranker — to go higher, improve retrieval,
not the reranker.

## Notes / knobs

- Teacher: `--teacher` (any `AutoModelForSequenceClassification` reranker; a
  larger teacher can raise the distillation ceiling at higher GPU cost).
- Student base: `train_distill.py --base-model`.
- Data scale: `prepare_distill_data.py --mmarco-queries / --negs-per-query`;
  `pairs_to_triplets.py --negs-per-query`.
- Training: `--epochs`, `--batch-size`, `--lr`, `--max-length`, `--max-train`.
- Export: `--no-quantize` to keep fp32.
- RunPod: `information/runpod-distill.sh` runs the full GPU pipeline with
  skip-guards; see `information/runpod-session.md` for the session log and the
  one-step pod re-launch.

### Simpler alternative: pointwise BCE

`training/train.py` fine-tunes a `CrossEncoder` directly on labeled pairs with
binary cross-entropy (no teacher). It is simpler and needs no GPU teacher pass,
but scored slightly below in-domain distillation on MIRACL (0.692 vs 0.701).
