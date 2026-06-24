"""Fine-tune a multilingual cross-encoder into an Indonesian reranker.

Uses sentence-transformers v5 ``CrossEncoderTrainer`` with binary
cross-entropy over (query, passage, label) pairs produced by prepare_data.py.

Defaults are tuned for a quick CPU smoke run; pass larger values (and use a GPU)
for a real run.

Run (venv with training deps active):
    python training/train.py --epochs 1 --batch-size 16 --max-train 3000
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

from common import read_jsonl

warnings.filterwarnings("ignore")


def _resolve_device_settings(args) -> tuple[bool, bool, bool, int]:
    import torch

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"CUDA device: {name} ({vram_gb:.1f} GiB VRAM)")
    else:
        print("WARNING: no CUDA device — training on CPU")

    fp16 = args.fp16
    bf16 = args.bf16
    if fp16 is None and bf16 is None:
        # Ampere+ (RTX 3090): bf16 is faster and stable for CE training.
        bf16 = use_cuda and torch.cuda.is_bf16_supported()
        fp16 = False
    elif fp16 is None:
        fp16 = False
    elif bf16 is None:
        bf16 = False

    workers = args.dataloader_workers
    if workers < 0:
        workers = min(8, os.cpu_count() or 4) if use_cuda else 0

    return use_cuda, fp16, bf16, workers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    parser.add_argument("--train-file", default="data/train.jsonl")
    parser.add_argument("--out-dir", default="models/ft-id-ce-v2")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--max-train", type=int, default=0, help="cap #pairs (0 = all)")
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="mixed precision fp16 (default: off; bf16 preferred on GPU)",
    )
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="mixed precision bf16 (default: on when CUDA supports it)",
    )
    parser.add_argument(
        "--dataloader-workers",
        type=int,
        default=-1,
        help="DataLoader workers (-1 = auto: 8 on GPU, 0 on CPU)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Accumulate gradients to simulate larger batch without OOM",
    )
    args = parser.parse_args()

    use_cuda, fp16, bf16, workers = _resolve_device_settings(args)
    if use_cuda:
        print(f"Training precision: fp16={fp16} bf16={bf16} batch={args.batch_size} "
              f"grad_accum={args.gradient_accumulation_steps} workers={workers}")

    from datasets import Dataset
    from sentence_transformers.cross_encoder import (
        CrossEncoder,
        CrossEncoderTrainer,
        CrossEncoderTrainingArguments,
    )
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    pairs = read_jsonl(args.train_file)
    if args.max_train and len(pairs) > args.max_train:
        pairs = pairs[: args.max_train]
    print(f"Training on {len(pairs)} pairs from {args.train_file}")

    train_ds = Dataset.from_dict(
        {
            "query": [p["query"] for p in pairs],
            "passage": [p["passage"] for p in pairs],
            "label": [float(p["label"]) for p in pairs],
        }
    )

    model = CrossEncoder(args.base_model, num_labels=1, max_length=args.max_length)
    loss = BinaryCrossEntropyLoss(model)

    train_args = CrossEncoderTrainingArguments(
        output_dir=str(Path(args.out_dir) / "_checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        fp16=fp16,
        bf16=bf16,
        logging_steps=20,
        save_strategy="no",
        report_to=[],
        dataloader_num_workers=workers,
        dataloader_pin_memory=use_cuda,
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        loss=loss,
    )
    trainer.train()

    model.save_pretrained(args.out_dir)
    print(f"Saved fine-tuned model -> {args.out_dir}")


if __name__ == "__main__":
    main()
