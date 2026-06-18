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
metrics:
- mrr
- ndcg
---

# rerank-indonesia

A lightweight **Indonesian (Bahasa Indonesia) cross-encoder reranker**, fine-tuned
from [`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)
on Indonesian query/passage pairs (TyDi QA Gold Passage + mined hard negatives).
It is small and CPU-friendly, so it runs fast even on a cheap VPS.

Built as part of [flashIndorank](https://github.com/madebyaris/flashIndorank).

## Evaluation

Held-out Indonesian eval (200 queries, 1 positive + 9 hard negatives each):

| model | top-1 | MRR | nDCG@10 |
| --- | --- | --- | --- |
| `ms-marco-MiniLM-L-12-v2` (English) | 0.615 | 0.743 | 0.805 |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (base) | 0.860 | 0.921 | 0.941 |
| **this model** | **0.895** | **0.940** | **0.956** |
| **this model (int8 ONNX)** | **0.895** | **0.940** | **0.955** |

## Usage

### sentence-transformers

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("madebyaris/rerank-indonesia")
query = "Bagaimana cara menurunkan berat badan?"
passages = [
    "Olahraga teratur dan pola makan sehat membantu mengurangi bobot tubuh.",
    "Harga emas global naik tajam dalam sepekan terakhir.",
]
scores = model.predict([[query, p] for p in passages])
print(scores)
```

### Lightweight ONNX (int8) via flashIndorank

The quantized ONNX model lives under `onnx/`. Download it and serve with
flashIndorank's `CustomReranker` (no torch/transformers needed at inference):

```python
from huggingface_hub import snapshot_download
from flashindorank import CustomReranker
from flashrank import RerankRequest

path = snapshot_download("madebyaris/rerank-indonesia", allow_patterns=["onnx/*"])
ranker = CustomReranker(f"{path}/onnx")
out = ranker.rerank(RerankRequest(
    query="Bagaimana cara menurunkan berat badan?",
    passages=[{"id": 1, "text": "Olahraga teratur dan pola makan sehat membantu mengurangi bobot tubuh."}],
))
print(out)
```

## Training

- Base: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- Data: Indonesian rows of TyDi QA (Gold Passage) + lexical hard negatives
- Loss: binary cross-entropy (sentence-transformers `CrossEncoderTrainer`)

See the [training pipeline](https://github.com/madebyaris/flashIndorank/blob/main/TRAINING.md).

## License

Apache-2.0, inherited from the base model. TyDi QA is Apache-2.0.
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
    parser.add_argument("--st-dir", default="models/ft-id-ce")
    parser.add_argument("--onnx-dir", default="models/ft-id-ce-onnx")
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
    print(f"Creating/ensuring repo {args.repo_id} ...")
    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True, private=args.private)

    # 1) CrossEncoder (PyTorch) at repo root; skip checkpoints and the
    #    auto-generated README (we upload our own model card).
    print("Uploading CrossEncoder weights + tokenizer ...")
    api.upload_folder(
        repo_id=args.repo_id,
        folder_path=args.st_dir,
        ignore_patterns=["_checkpoints/*", "README.md"],
        commit_message="Add fine-tuned Indonesian CrossEncoder",
    )

    # 2) Quantized ONNX under onnx/ (model.onnx is the int8 build).
    print("Uploading quantized ONNX + tokenizer under onnx/ ...")
    api.upload_folder(
        repo_id=args.repo_id,
        folder_path=args.onnx_dir,
        path_in_repo="onnx",
        ignore_patterns=["model_quantized.onnx"],
        commit_message="Add quantized ONNX model",
    )

    # 3) Model card.
    print("Uploading model card ...")
    with tempfile.TemporaryDirectory() as tmp:
        card = Path(tmp) / "README.md"
        card.write_text(MODEL_CARD, encoding="utf-8")
        api.upload_file(
            path_or_fileobj=str(card),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            commit_message="Add model card",
        )

    print(f"Done: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
