"""Distill a strong teacher reranker into a tiny Indonesian cross-encoder.

Margin-MSE knowledge distillation (Hofstätter et al.): the small student learns
the teacher's score *margins* between positive/negative passages. This yields
near-teacher ranking quality at a fraction of the size — ideal for serving a
quantized ONNX reranker on a cheap VPS.

Inputs:
* ``--train-file``  distill_train.jsonl: {"query","positive","negative","label"}
                    where label = teacher(q,pos) - teacher(q,neg)
* ``--eval-file``   distill_eval.jsonl:  {"query","positives":[...],"negatives":[...]}
                    used by a CrossEncoderRerankingEvaluator for monitoring

Run (GPU):
    python training/train_distill.py --epochs 2 --batch-size 64
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

from common import read_jsonl

warnings.filterwarnings("ignore")


def _device_settings(args):
    import torch

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"CUDA device: {name} ({vram:.1f} GiB VRAM)", flush=True)
    else:
        print("WARNING: no CUDA — distillation on CPU will be slow", flush=True)
    bf16 = use_cuda and torch.cuda.is_bf16_supported()
    workers = min(8, os.cpu_count() or 4) if use_cuda else 0
    return use_cuda, bf16, workers


def _build_evaluator(eval_file: str, max_neg: int):
    from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator

    rows = read_jsonl(eval_file)
    samples = [
        {
            "query": r["query"],
            "positive": list(r.get("positives") or []),
            "negative": list(r.get("negatives") or [])[:max_neg],
        }
        for r in rows
        if r.get("positives") and r.get("negatives")
    ]
    print(f"Reranking evaluator: {len(samples)} queries from {eval_file}", flush=True)
    return CrossEncoderRerankingEvaluator(samples=samples, name="mmarco-id-dev", at_k=10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
                        help="student init (small, multilingual reranker)")
    parser.add_argument("--train-file", default="data/distill_train.jsonl")
    parser.add_argument("--eval-file", default="data/distill_eval.jsonl")
    parser.add_argument("--out-dir", default="models/ft-id-ce-distill")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=8e-6, help="low LR is standard for distillation")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--eval-steps", type=int, default=2000)
    parser.add_argument("--eval-max-neg", type=int, default=30)
    parser.add_argument("--max-train", type=int, default=0, help="cap #triplets (0 = all)")
    args = parser.parse_args()

    use_cuda, bf16, workers = _device_settings(args)
    print(f"Distill: bf16={bf16} batch={args.batch_size} lr={args.lr} "
          f"epochs={args.epochs} workers={workers}", flush=True)

    from datasets import Dataset
    from sentence_transformers.cross_encoder import (
        CrossEncoder,
        CrossEncoderTrainer,
        CrossEncoderTrainingArguments,
    )
    from sentence_transformers.cross_encoder.losses import MarginMSELoss

    triplets = read_jsonl(args.train_file)
    if args.max_train and len(triplets) > args.max_train:
        triplets = triplets[: args.max_train]
    print(f"Training on {len(triplets)} scored triplets from {args.train_file}", flush=True)

    # Column order (query, positive, negative) feeds the model; "label" = teacher margin.
    train_ds = Dataset.from_dict(
        {
            "query": [t["query"] for t in triplets],
            "positive": [t["positive"] for t in triplets],
            "negative": [t["negative"] for t in triplets],
            "label": [float(t["label"]) for t in triplets],
        }
    )

    model = CrossEncoder(args.base_model, num_labels=1, max_length=args.max_length)
    loss = MarginMSELoss(model)

    evaluator = None
    if Path(args.eval_file).exists():
        try:
            evaluator = _build_evaluator(args.eval_file, args.eval_max_neg)
        except Exception as exc:  # noqa: BLE001
            print(f"Evaluator disabled ({type(exc).__name__}: {exc})", flush=True)

    train_args = CrossEncoderTrainingArguments(
        output_dir=str(Path(args.out_dir) / "_checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        bf16=bf16,
        fp16=False,
        logging_steps=200,
        eval_strategy="steps" if evaluator is not None else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        report_to=[],
        dataloader_num_workers=workers,
        dataloader_pin_memory=use_cuda,
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        loss=loss,
        evaluator=evaluator,
    )
    if evaluator is not None:
        print("Baseline (pre-distill) reranking metrics:", flush=True)
        try:
            print(evaluator(model), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  baseline eval skipped ({type(exc).__name__}: {exc})", flush=True)

    trainer.train()

    model.save_pretrained(args.out_dir)
    print(f"Saved distilled student -> {args.out_dir}", flush=True)
    if evaluator is not None:
        print("Final reranking metrics:", flush=True)
        try:
            print(evaluator(model), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  final eval skipped ({type(exc).__name__}: {exc})", flush=True)


if __name__ == "__main__":
    main()
