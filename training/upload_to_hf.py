"""Push the fine-tuned Indonesian reranker to the Hugging Face Hub.

Uploads:
* the sentence-transformers CrossEncoder (PyTorch/safetensors) to the repo root,
  so `CrossEncoder("<repo>")` works out of the box, and
* the quantized ONNX model + tokenizer under ``onnx/`` for lightweight serving
  (e.g. flashIndorank's CustomReranker),
* a generated model card (README.md).

Auth: set a HF write token in the environment as ``HF_TOKEN`` (or
``HUGGING_FACE_HUB_TOKEN``). Create one at https://huggingface.co/settings/tokens
with "Write" access.

Run:
    python training/upload_to_hf.py --repo-id madebyaris/rerank-indonesia
    python training/upload_to_hf.py --revision huggingface --st-dir models/ft-id-ce-v2
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

MODEL_CARD = """\
---
language:
- id
license: apache-2.0
library_name: sentence-transformers
pipeline_tag: text-ranking
tags:
- reranker
- cross-encoder
- text-ranking
- indonesian
- bahasa-indonesia
- flashrank
- onnx
base_model: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
datasets:
- google-research-datasets/tydiqa
- miracl/miracl
metrics:
- mrr
- ndcg
---

# rerank-indonesia (v2 preview)

A lightweight **Indonesian (Bahasa Indonesia) cross-encoder reranker**, fine-tuned
from [`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)
on Indonesian query/passage pairs from **TyDi QA** and **MIRACL-id**, with
**BM25 + dense hard-negative mining** (15 negatives per query).

> **Branch note:** This `huggingface` branch holds the **v2 preview** checkpoint.
> The `main` branch still serves the original v1 model. Full training on all
> 162k pairs is intended to run on RunPod GPU — see
> [flashIndorank TRAINING.md](https://github.com/madebyaris/flashIndorank/blob/main/TRAINING.md).

Built as part of [flashIndorank](https://github.com/madebyaris/flashIndorank).

## Evaluation

TyDi holdout (200 queries, 1 positive + 9 hard negatives each):

| model | top-1 | MRR | nDCG@10 |
| --- | --- | --- | --- |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (base) | 0.905 | 0.943 | 0.957 |
| **this model (v2 preview)** | **0.925** | **0.953** | **0.964** |
| v1 on `main` branch | 0.915 | 0.953 | 0.965 |

MIRACL-id official rerank eval (BM25 top-100 → rerank): run
`python training/eval_miracl.py --model madebyaris/rerank-indonesia --revision huggingface`

## Usage

### sentence-transformers

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("madebyaris/rerank-indonesia", revision="huggingface")
query = "Bagaimana cara menurunkan berat badan?"
passages = [
    "Olahraga teratur dan pola makan sehat membantu mengurangi bobot tubuh.",
    "Harga emas global naik tajam dalam sepekan terakhir.",
]
scores = model.predict([[query, p] for p in passages])
print(scores)
```

### Lightweight ONNX (int8) via flashIndorank

```python
from huggingface_hub import snapshot_download
from flashindorank import CustomReranker
from flashrank import RerankRequest

path = snapshot_download(
    "madebyaris/rerank-indonesia",
    revision="huggingface",
    allow_patterns=["onnx/*"],
)
ranker = CustomReranker(f"{path}/onnx")
out = ranker.rerank(RerankRequest(
    query="Bagaimana cara menurunkan berat badan?",
    passages=[{"id": 1, "text": "Olahraga teratur dan pola makan sehat membantu mengurangi bobot tubuh."}],
))
print(out)
```

## Training

- Base: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- Data: TyDi QA Indonesian + MIRACL-id train (BM25 hard negatives)
- Loss: binary cross-entropy (`CrossEncoderTrainer`)

See [TRAINING.md](https://github.com/madebyaris/flashIndorank/blob/main/TRAINING.md).

## License

Apache-2.0, inherited from the base model. TyDi QA and MIRACL are Apache-2.0.
"""


def _get_token() -> str | None:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="madebyaris/rerank-indonesia")
    parser.add_argument("--revision", default=None, help="HF branch/tag (e.g. huggingface)")
    parser.add_argument("--st-dir", default="models/ft-id-ce-v2")
    parser.add_argument("--onnx-dir", default="models/ft-id-ce-v2-onnx")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    token = _get_token()
    if not token:
        sys.exit(
            "No Hugging Face token found. Set HF_TOKEN (write access) in the "
            "environment, e.g. export HF_TOKEN=hf_xxx, then re-run."
        )

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    branch = args.revision or "main"
    print(f"Creating/ensuring repo {args.repo_id} (branch: {branch}) ...")
    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True, private=args.private)

    upload_kwargs = (
        {"repo_id": args.repo_id, "revision": args.revision}
        if args.revision
        else {"repo_id": args.repo_id}
    )

    # 1) CrossEncoder (PyTorch) at repo root; skip checkpoints and the
    #    auto-generated README (we upload our own model card).
    print(f"Uploading CrossEncoder weights + tokenizer -> {branch} ...")
    api.upload_folder(
        **upload_kwargs,
        folder_path=args.st_dir,
        ignore_patterns=["_checkpoints/*", "README.md"],
        commit_message=f"Add Indonesian CrossEncoder v2 ({branch})",
    )

    # 2) Quantized ONNX under onnx/ (model.onnx is the int8 build).
    print(f"Uploading quantized ONNX + tokenizer under onnx/ -> {branch} ...")
    api.upload_folder(
        **upload_kwargs,
        folder_path=args.onnx_dir,
        path_in_repo="onnx",
        ignore_patterns=["model_quantized.onnx"],
        commit_message=f"Add quantized ONNX v2 ({branch})",
    )

    # 3) Model card.
    print(f"Uploading model card -> {branch} ...")
    with tempfile.TemporaryDirectory() as tmp:
        card = Path(tmp) / "README.md"
        card.write_text(MODEL_CARD, encoding="utf-8")
        api.upload_file(
            **upload_kwargs,
            path_or_fileobj=str(card),
            path_in_repo="README.md",
            commit_message=f"Add model card v2 ({branch})",
        )

    url = f"https://huggingface.co/{args.repo_id}"
    if args.revision:
        url += f"/tree/{args.revision}"
    print(f"Done: {url}")


if __name__ == "__main__":
    main()
