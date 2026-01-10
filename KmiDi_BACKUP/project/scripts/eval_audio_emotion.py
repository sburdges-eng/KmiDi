#!/usr/bin/env python3
"""
Lightweight evaluation for audio emotion classification.

Inputs:
  - Ground truth manifest CSV (e.g., datasets/validation/audio_emotion_manifest.csv)
    Expected columns: id | file_path | emotion_label | split (others ignored)
  - Predictions JSONL: one object per line:
      {
        "id": "clip_id",
        "pred_label": "happy",
        "probs": {"happy": 0.7, "sad": 0.2, "angry": 0.1}  # optional but needed for ECE
      }

Outputs:
  - Prints macro F1, per-class F1, Cohen's kappa, and ECE (10-bin, if probs provided)
  - Writes metrics to <output_dir>/metrics.json

No external dependencies beyond the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class Example:
    id: str
    label: str
    split: str


def load_manifest(path: Path, split: Optional[str]) -> Dict[str, Example]:
    examples: Dict[str, Example] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_split = row.get("split", "").strip()
            if split and row_split != split:
                continue
            ex_id = row.get("id") or row.get("file_path") or ""
            label = row.get("emotion_label") or row.get("label") or ""
            if not ex_id or not label:
                continue
            examples[ex_id] = Example(id=ex_id, label=label.strip(), split=row_split)
    return examples


def load_predictions(path: Path) -> List[Dict[str, object]]:
    preds: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            preds.append(json.loads(line))
    return preds


def compute_confusion(truth: Dict[str, Example], preds: List[Dict[str, object]]) -> Tuple[Dict[str, Dict[str, int]], List[Tuple[float, bool]]]:
    conf: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    prob_records: List[Tuple[float, bool]] = []

    for p in preds:
        ex_id = p.get("id")
        pred_label = p.get("pred_label")
        if not ex_id or ex_id not in truth or not pred_label:
            continue
        true_label = truth[ex_id].label
        conf[true_label][pred_label] += 1

        probs = p.get("probs")
        if isinstance(probs, dict) and len(probs) > 0:
            prob_pred = max(probs.items(), key=lambda kv: kv[1])
            prob_records.append((float(prob_pred[1]), prob_pred[0] == true_label))

    return conf, prob_records


def f1_scores(conf: Dict[str, Dict[str, int]]) -> Tuple[Dict[str, float], float]:
    labels = sorted({*conf.keys(), *{pl for tl in conf for pl in conf[tl].keys()}})
    per_class: Dict[str, float] = {}
    f1_sum = 0.0
    for label in labels:
        tp = conf[label].get(label, 0)
        fp = sum(conf[tl].get(label, 0) for tl in labels if tl != label)
        fn = sum(conf[label].get(pl, 0) for pl in labels if pl != label)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[label] = f1
        f1_sum += f1

    macro_f1 = f1_sum / len(labels) if labels else 0.0
    return per_class, macro_f1


def cohen_kappa(conf: Dict[str, Dict[str, int]]) -> float:
    labels = sorted({*conf.keys(), *{pl for tl in conf for pl in conf[tl].keys()}})
    total = sum(conf[tl][pl] for tl in labels for pl in labels)
    if total == 0:
        return 0.0

    po = sum(conf[l].get(l, 0) for l in labels) / total
    row_marginals = {l: sum(conf[l].get(pl, 0) for pl in labels) for l in labels}
    col_marginals = {l: sum(conf[tl].get(l, 0) for tl in labels) for l in labels}
    pe = sum(row_marginals[l] * col_marginals[l] for l in labels) / (total * total)
    denom = 1 - pe
    if denom == 0:
        return 0.0
    return (po - pe) / denom


def expected_calibration_error(prob_records: List[Tuple[float, bool]], n_bins: int = 10) -> float:
    if not prob_records:
        return 0.0
    bins = [0.0 for _ in range(n_bins)]
    bin_acc = [0.0 for _ in range(n_bins)]
    bin_count = [0 for _ in range(n_bins)]

    for prob, correct in prob_records:
        prob_clamped = max(0.0, min(1.0, prob))
        idx = min(int(prob_clamped * n_bins), n_bins - 1)
        bins[idx] += prob_clamped
        bin_acc[idx] += 1.0 if correct else 0.0
        bin_count[idx] += 1

    ece = 0.0
    total = len(prob_records)
    for i in range(n_bins):
        if bin_count[i] == 0:
            continue
        avg_conf = bins[i] / bin_count[i]
        avg_acc = bin_acc[i] / bin_count[i]
        ece += (bin_count[i] / total) * abs(avg_conf - avg_acc)
    return ece


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate audio emotion predictions without external deps.")
    parser.add_argument("--manifest", type=Path, default=Path("datasets/validation/audio_emotion_manifest.csv"), help="Ground truth manifest CSV.")
    parser.add_argument("--predictions", type=Path, required=True, help="Predictions JSONL file.")
    parser.add_argument("--split", type=str, default="val_gold", help="Split to evaluate (e.g., val_gold, train, val, test).")
    parser.add_argument("--output-dir", type=Path, default=Path("output/audio_emotion_eval"), help="Directory to write metrics.json.")
    args = parser.parse_args()

    truth = load_manifest(args.manifest, args.split)
    preds = load_predictions(args.predictions)

    conf, prob_records = compute_confusion(truth, preds)
    per_class_f1, macro_f1 = f1_scores(conf)
    kappa = cohen_kappa(conf)
    ece = expected_calibration_error(prob_records) if prob_records else None

    metrics = {
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "cohen_kappa": kappa,
    }
    if ece is not None:
        metrics["ece"] = ece

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Metrics written to {out_path}")


if __name__ == "__main__":
    main()
