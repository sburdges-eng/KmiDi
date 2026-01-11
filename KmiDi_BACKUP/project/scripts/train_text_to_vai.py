"""
Lightweight fine-tuning script for text→VAI (valence, arousal, intensity) regression.

Defaults target DistilRoBERTa. Designed to be data-path agnostic:
- EmoBank CSV via Hugging Face "emo_bank" or a local CSV with columns: text, valence, arousal, dominance/intensity.
- Exports an ONNX model if optimum/onnxruntime are installed.

Example:
  python scripts/train_text_to_vai.py \
      --dataset emo_bank \
      --output-dir models/text_to_vai \
      --export-onnx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def normalize_example(ex: dict) -> dict:
    """Normalize 1–5 VAD to [-1,1] valence/arousal, [0,1] intensity."""
    v = (float(ex["valence"]) - 3.0) / 2.0
    a = (float(ex["arousal"]) - 3.0) / 2.0
    # Dominance often 1–5; map to [0,1] as intensity proxy.
    d = (float(ex.get("dominance", ex.get("intensity", 3.0))) - 1.0) / 4.0
    ex["labels"] = [v, a, d]
    return ex


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    mse = np.mean((preds - labels) ** 2)
    return {"mse": mse}


def maybe_export_onnx(model_dir: Path, onnx_out: Path, opset: int = 17) -> None:
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification
    except Exception as exc:
        print(f"[warn] optimum/onnxruntime not available, skipping ONNX export: {exc}")
        return

    ort_model = ORTModelForSequenceClassification.from_pretrained(
        model_dir,
        file_name=onnx_out.name,
        export=True,
        opset=opset,
    )
    ort_model.save_pretrained(str(onnx_out.parent))
    print(f"[ok] ONNX exported to {onnx_out}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune text→VAI regression.")
    parser.add_argument("--dataset", default="emo_bank", help="HF dataset name or local CSV path.")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--valence-column", default="valence")
    parser.add_argument("--arousal-column", default="arousal")
    parser.add_argument("--dominance-column", default="dominance")
    parser.add_argument("--model", default="distilroberta-base")
    parser.add_argument("--output-dir", default="models/text_to_vai")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "emo_bank":
        ds = load_dataset("emo_bank")
    else:
        ds = load_dataset("csv", data_files={"train": args.dataset})

    def rename_cols(ex):
        ex["text"] = ex.get(args.text_column, ex.get("text"))
        ex["valence"] = ex.get(args.valence_column, ex.get("valence"))
        ex["arousal"] = ex.get(args.arousal_column, ex.get("arousal"))
        ex["dominance"] = ex.get(args.dominance_column, ex.get("dominance", 3.0))
        return ex

    ds = ds.map(rename_cols).map(normalize_example)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "right"

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )

    tokenized = ds.map(tokenize_fn, batched=True)
    tokenized = tokenized.rename_column("labels", "labels")
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3)

    train_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized.get("train"),
        eval_dataset=tokenized.get("validation") or tokenized.get("test"),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    if args.export_onnx:
        maybe_export_onnx(output_dir, output_dir / "model.onnx")

    meta = {
        "model": args.model,
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "max_length": args.max_length,
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[ok] Finished. Saved to {output_dir}")


if __name__ == "__main__":
    main()
