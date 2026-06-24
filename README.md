# flashIndorank

Lightweight, high-performance **cross-encoder reranker** built on
[FlashRank](https://github.com/PrithivirajDamodaran/FlashRank). It is tuned to
run **fast on cheap VPS hardware** (1–2 vCPU, little RAM) by defaulting to tiny
ONNX models and offering a two-stage **cascade** that gets close to
strong-model quality at a fraction of the cost.

## Why

Reranking dramatically improves search/RAG quality, but most rerankers are too
heavy for small boxes. flashIndorank keeps things small and fast:

- **Tiny defaults** – `ms-marco-TinyBERT-L-2-v2` (3.3 MB) for the fast path,
  `ms-marco-MiniLM-L-12-v2` (21.6 MB, quantized) for the strong path.
- **Cascade reranking** – a tiny model prunes the candidate set, then a stronger
  model re-scores only the survivors. Strong quality, low cost.
- **Warm singletons** – each model's ONNX session is loaded once per process.
- **Thread control** – pin ONNX Runtime threads so it doesn't thrash a small VPS.
- **No GPU required** – pure CPU ONNX inference.

## Indonesian (Bahasa Indonesia) support

Important: the default models (`ms-marco-TinyBERT-L-2-v2`, `ms-marco-MiniLM-L-12-v2`)
are trained on **English** MS-MARCO. They lean heavily on lexical overlap and are
**not strong on Indonesian semantics**. FlashRank's bundled multilingual model,
`ms-marco-MultiBERT-L-12` (100+ languages incl. Indonesian), is the right choice
for Bahasa Indonesia, though on a hard paraphrase eval it is only modestly better
and is heavier (~99 MB vs 3–22 MB).

Reproduce the evidence yourself:

```bash
python benchmarks/eval_indonesian.py
```

Example output (paraphrased relevant passage, low lexical overlap):

| model | top-1 acc | MRR |
| --- | --- | --- |
| ms-marco-TinyBERT-L-2-v2 | 0.40 | 0.633 |
| ms-marco-MiniLM-L-12-v2 | 0.40 | 0.633 |
| ms-marco-MultiBERT-L-12 (multilingual) | 0.40 | 0.667 |

To make the multilingual model the default everywhere (no code change), set:

```bash
export FLASHINDORANK_DEFAULT_MODEL=ms-marco-MultiBERT-L-12
```

…or pass `"model": "ms-marco-MultiBERT-L-12"` per request.

### Ready-to-use Indonesian reranker (recommended)

We publish a tiny, VPS-friendly Indonesian reranker:
**[`madebyaris/rerank-indonesia`](https://huggingface.co/madebyaris/rerank-indonesia)**.

It is built by **Margin-MSE knowledge distillation**: a 568M teacher
(`BAAI/bge-reranker-v2-m3`) supervises the tiny `mmarco-mMiniLMv2-L12-H384`
student on in-domain Indonesian data (TyDi QA + MIRACL-id). The student stays
small (**118 MB int8 ONNX**) yet reaches ~98% of the teacher's ranking quality.

**MIRACL-id** dev, official retrieve-then-rerank protocol (BM25 top-100 →
rerank, 960 queries, `pytrec_eval`):

| model | size | nDCG@10 | MRR@10 |
| --- | --- | --- | --- |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (base) | tiny | 0.656 | 0.623 |
| **`madebyaris/rerank-indonesia`** (distilled) | tiny | **0.701** | **0.677** |
| `BAAI/bge-reranker-v2-m3` (teacher) | 568M | 0.712 | 0.689 |

The distilled model improves nDCG@10 by **+4.5 points** over the base while
staying within ~1 point of the 17× larger teacher. (Recall@100 = 0.760 is the
BM25 first-stage ceiling and bounds every reranker.)

Use it with sentence-transformers:

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("madebyaris/rerank-indonesia")
scores = model.predict([[query, p] for p in passages])
```

…or serve the lightweight int8 ONNX inside flashIndorank:

```bash
huggingface-cli download madebyaris/rerank-indonesia --include "onnx/*" \
  --local-dir models/rerank-indonesia
export FLASHINDORANK_CUSTOM_MODELS="id-reranker=$PWD/models/rerank-indonesia/onnx"
python -m flashindorank
# POST /rerank with {"model": "id-reranker", ...}
```

### Train or improve it yourself

To reproduce or push further, see [`TRAINING.md`](TRAINING.md): build in-domain
(query, positive, negative) triplets → score them with a strong teacher
(Margin-MSE labels) → distill into the tiny student → export quantized ONNX.

## Models

| Model | Size | Notes |
| --- | --- | --- |
| `ms-marco-TinyBERT-L-2-v2` | 3.3 MB | Fastest. Default fast / first stage. |
| `ms-marco-MiniLM-L-12-v2` | 21.6 MB | Stronger, still light. Default strong / second stage. |
| `ms-marco-MultiBERT-L-12` | ~150 MB | Multilingual (incl. Indonesian). |
| `ce-esci-MiniLM-L12-v2` | ~22 MB | Product / e-commerce search. |
| `rank-T5-flan` | ~110 MB | Strongest ONNX option, heaviest. |

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # or: pip install -e ".[dev]"
```

## Run the API

```bash
python -m flashindorank
# serves on http://0.0.0.0:8000 (interactive docs at /docs)
```

Useful environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `FLASHINDORANK_CACHE_DIR` | `~/.cache/flashindorank` | Where ONNX model weights are cached. |
| `FLASHINDORANK_DEFAULT_MODEL` | `ms-marco-TinyBERT-L-2-v2` | Default model when a request omits one (set to `ms-marco-MultiBERT-L-12` for Indonesian). |
| `FLASHINDORANK_DEFAULT_STRONG_MODEL` | `ms-marco-MiniLM-L-12-v2` | Default cascade second-stage model. |
| `FLASHINDORANK_PRELOAD_MODELS` | _(none)_ | Comma-separated models to load at startup. |
| `FLASHINDORANK_CUSTOM_MODELS` | _(none)_ | Local ONNX rerankers as `name=/path,name2=/path2` (e.g. a fine-tuned Indonesian model). |
| `FLASHINDORANK_INTRA_OP_THREADS` | `0` (auto) | ONNX intra-op threads (pin to 1–2 on tiny VPS). |
| `FLASHINDORANK_INTER_OP_THREADS` | `0` (auto) | ONNX inter-op threads. |
| `FLASHINDORANK_MAX_LENGTH` | `512` | Max token length. |
| `FLASHINDORANK_PORT` | `8000` | HTTP port. |

### Endpoints

- `GET /health` – status + loaded models.
- `GET /models` – available models with metadata.
- `POST /rerank` – single-model rerank.
- `POST /rerank/cascade` – two-stage cascade rerank.

```bash
curl -X POST http://localhost:8000/rerank -H 'Content-Type: application/json' -d '{
  "query": "What is the capital of Indonesia?",
  "passages": [
    {"id": 1, "text": "Bali is a popular tourist island in Indonesia."},
    {"id": 2, "text": "Jakarta is the capital and largest city of Indonesia."}
  ],
  "top_k": 1
}'
```

`passages` may be plain strings or `{"id", "text", "meta"}` objects.

## Use as a library

```python
from flashindorank import rerank, rerank_cascade

results = rerank("what is a panda?", ["a giant panda", "a laptop"], top_k=1)

# Strong but cheap: tiny model prunes to top 50, MiniLM re-scores those.
results = rerank_cascade(query, passages, prune_to=50, top_k=10)
```

## Benchmark

```bash
python benchmarks/benchmark.py --passages 100 --runs 30 --prune-to 20
```

Example (CPU, 100 passages): TinyBERT ~13 ms, MiniLM ~126 ms, cascade ~77 ms —
the cascade keeps strong-model quality while cutting strong-model cost.

## Develop

```bash
pip install -e ".[dev]"
ruff check src benchmarks tests
pytest
```
