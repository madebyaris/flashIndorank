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
- knowledge-distillation
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

# rerank-indonesia

A lightweight **Indonesian (Bahasa Indonesia) cross-encoder reranker**, small
enough to serve on a cheap CPU VPS yet competitive with a 17× larger model.

It is built by **Margin-MSE knowledge distillation**: a strong multilingual
teacher, [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3)
(568M params), supervises the tiny student
[`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)
on in-domain Indonesian (query, positive, negative) triplets from **TyDi QA** and
**MIRACL-id** (with BM25 + dense hard-negative mining). The student learns the
teacher's score *margin* between relevant and non-relevant passages.

Built as part of [flashIndorank](https://github.com/madebyaris/flashIndorank).

## Evaluation

**MIRACL-id** official retrieve-then-rerank protocol (BM25 top-100 → rerank,
960 dev queries, `pytrec_eval`):

| model | params | nDCG@10 | MRR@10 | Recall@100 |
| --- | --- | --- | --- | --- |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (base) | tiny | 0.656 | 0.623 | 0.760 |
| **this model** (in-domain distillation) | **tiny** | **0.701** | **0.677** | 0.760 |
| `BAAI/bge-reranker-v2-m3` (teacher) | 568M | 0.712 | 0.689 | 0.760 |

The distilled student improves nDCG@10 by **+4.5 points** over the base while
staying within **~1 point of the 568M teacher** — roughly 98% of the teacher's
ranking quality at a fraction of the size and latency. (Recall@100 is the BM25
first-stage ceiling and bounds all rerankers.)

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

- Student / base: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- Teacher: `BAAI/bge-reranker-v2-m3`
- Method: Margin-MSE knowledge distillation (Hofstätter et al., 2020) —
  `label = teacher(q, pos) - teacher(q, neg)`
- Data: in-domain Indonesian triplets from TyDi QA + MIRACL-id train,
  BM25 + dense hard negatives
- Optimizer: 3 epochs, lr 8e-6, bf16, `MarginMSELoss` (`CrossEncoderTrainer`)

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

    from huggingface_hub.utils import HfHubHTTPError

    api = HfApi(token=token)
    branch = args.revision or "main"
    print(f"Creating/ensuring repo {args.repo_id} (branch: {branch}) ...")
    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True, private=args.private)

    if args.revision and args.revision != "main":
        try:
            api.create_branch(
                repo_id=args.repo_id,
                branch=args.revision,
                repo_type="model",
                revision="main",
            )
            print(f"Created branch {args.revision!r} from main")
        except HfHubHTTPError as exc:
            if "already exists" not in str(exc).lower():
                raise
            print(f"Branch {args.revision!r} already exists")

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
