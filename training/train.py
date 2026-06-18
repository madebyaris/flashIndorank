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
import warnings
from pathlib import Path

from common import read_jsonl

warnings.filterwarnings("ignore")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    parser.add_argument("--train-file", default="data/train.jsonl")
    parser.add_argument("--out-dir", default="models/ft-id-ce")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--max-train", type=int, default=0, help="cap #pairs (0 = all)")
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    args = parser.parse_args()

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
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        fp16=False,  # CPU
        bf16=False,
        logging_steps=20,
        save_strategy="no",
        report_to=[],
        dataloader_num_workers=0,
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
