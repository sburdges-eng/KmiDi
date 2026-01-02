#!/usr/bin/env python3
"""
Generate prediction JSONL from the unified emotion manifest for evaluation.

Modes:
- Baseline (no torch required): majority-class predictor based on manifest labels.
- Torch model (optional): if --checkpoint is provided and torch is available,
  loads a torchscript model and runs inference on audio using torchaudio.
  Provide --class-names to map logits -> labels; script asserts the count matches.

Outputs:
- JSONL written to --output (default: output/audio_emotion_eval/predictions.jsonl)
  Each line: {"id": "...", "pred_label": "...", "probs": {"label": prob, ...}}
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import torch  # type: ignore
    import torchaudio  # type: ignore
except Exception:
    torch = None  # type: ignore
    torchaudio = None  # type: ignore

from penta_core.ml.datasets.unified_emotion import UnifiedEmotionDataset


def majority_baseline(labels: List[str]) -> Dict[str, float]:
    counts = Counter(labels)
    total = sum(counts.values())
    if total == 0:
        return {}
    # Probability mass on majority; uniform over others to keep a distribution
    majority_label, majority_count = counts.most_common(1)[0]
    remaining = total - majority_count
    probs = {}
    for label, count in counts.items():
        if label == majority_label:
            probs[label] = majority_count / total
        else:
            probs[label] = (count / total) * 0.5  # dampened minor classes
    # Renormalize
    s = sum(probs.values())
    if s:
        probs = {k: v / s for k, v in probs.items()}
    return probs


def run_baseline(dataset: UnifiedEmotionDataset) -> List[Dict[str, Any]]:
    labels = [f"{item['dataset']}_{item['id']}" for item in dataset.items]  # unused but keeps structure
    true_labels = [item.get("label") or item.get("emotion_label") or "" for item in dataset.items]
    # Fall back to dataset name if no explicit label; this keeps output sane for placeholder manifests
    fallback_labels = [tl if tl else item.get("dataset", "unknown") for tl, item in zip(true_labels, dataset.items)]
    probs = majority_baseline([l for l in fallback_labels if l])
    if not probs:
        probs = {"unknown": 1.0}
    pred_label = max(probs.items(), key=lambda kv: kv[1])[0]
    return [
        {"id": item["id"], "pred_label": pred_label, "probs": probs}
        for item in dataset.items
    ]


def run_torch_inference(dataset: UnifiedEmotionDataset, checkpoint: Path, class_names: List[str]) -> List[Dict[str, Any]]:
    if torch is None or torchaudio is None:
        raise RuntimeError("torch/torchaudio not available; cannot run torch inference.")

    # Placeholder: expecting a TorchScript model that takes waveform tensor and returns logits
    model = torch.jit.load(str(checkpoint))
    model.eval()

    labels = class_names
    label_to_idx = {l: i for i, l in enumerate(labels)}

    preds: List[Dict[str, Any]] = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            batch = dataset[idx]
            wav = batch["audio"]
            logits = model(wav.unsqueeze(0))  # adapt for your model signature
            if logits.shape[-1] != len(labels):
                raise RuntimeError(f"Model logits dim {logits.shape[-1]} does not match number of class names {len(labels)}")
            probs_tensor = torch.softmax(logits, dim=-1)[0]
            probs = {label: float(probs_tensor[label_to_idx[label]].item()) for label in labels}
            pred_label = max(probs.items(), key=lambda kv: kv[1])[0]
            preds.append({
                "id": batch["id"],
                "pred_label": pred_label,
                "probs": probs,
            })
    return preds


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate predictions JSONL from unified emotion manifest.")
    parser.add_argument("--manifest", type=Path, default=Path("datasets/validation/emotion_manifest.json"), help="Unified manifest JSON.")
    parser.add_argument("--split", type=str, default=None, help="Optional split filter (train|val|test).")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional TorchScript model for real inference.")
    parser.add_argument("--class-names", type=str, default=None, help="Comma-separated class names matching model logits.")
    parser.add_argument("--output", type=Path, default=Path("output/audio_emotion_eval/predictions.jsonl"), help="Output JSONL path.")
    args = parser.parse_args()

    dataset = UnifiedEmotionDataset(args.manifest, split=args.split)

    if args.checkpoint:
        if not args.class_names:
            raise SystemExit("When using --checkpoint you must provide --class-names comma-separated.")
        class_names = [c.strip() for c in args.class_names.split(",") if c.strip()]
        if not class_names:
            raise SystemExit("No class names parsed from --class-names.")
        preds = run_torch_inference(dataset, args.checkpoint, class_names)
    else:
        preds = run_baseline(dataset)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")

    print(f"Wrote {len(preds)} predictions to {args.output}")
    if not args.checkpoint:
        print("Note: baseline predictor used (no checkpoint provided).")


if __name__ == "__main__":
    main()
