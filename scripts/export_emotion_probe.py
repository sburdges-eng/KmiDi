#!/usr/bin/env python3
"""
Export trained EmotionProbe to ONNX for use in the C++ plugin.

Loads the best probe checkpoint (from train_emotion_probe.py), traces with
a dummy latent vector, and exports to ONNX opset 17 with fixed input/output
shapes. Verifies numerical parity (max_diff < 1e-5) and prints a model summary.

Usage:
    python scripts/export_emotion_probe.py
    python scripts/export_emotion_probe.py \\
        --checkpoint checkpoints/emotion_probe/best_probe.pt \\
        --output models/emotion_probe_v01.onnx
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch

from music_brain.jepa.emotion_probe import EmotionProbe

logger = logging.getLogger(__name__)

# Fixed I/O shapes (batch=1 for C++ plugin)
INPUT_SHAPE = (1, 256)   # (batch, latent_dim)
OUTPUT_SHAPE = (1, 2)    # (batch, [valence, arousal])


def load_probe(checkpoint_path: Path) -> tuple[EmotionProbe, dict]:
    """Load EmotionProbe from checkpoint. Returns (probe, checkpoint_dict)."""
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    latent_dim = ckpt.get("latent_dim", 256)
    hidden_dim = ckpt.get("hidden_dim", 128)
    probe = EmotionProbe(latent_dim=latent_dim, hidden_dim=hidden_dim)
    probe.load_state_dict(ckpt["probe"])
    probe.eval()
    logger.info(
        "Loaded probe: epoch=%d, val_loss=%.6f, latent_dim=%d, hidden_dim=%d",
        ckpt.get("epoch", -1), ckpt.get("val_loss", float("nan")), latent_dim, hidden_dim,
    )
    return probe, ckpt


def export_onnx(probe: EmotionProbe, output_path: Path) -> Path:
    """Export probe to ONNX (opset 17, fixed shapes)."""
    import onnx

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(*INPUT_SHAPE)

    torch.onnx.export(
        probe,
        dummy,
        str(output_path),
        opset_version=17,
        input_names=["latent"],
        output_names=["emotion"],
        dynamic_axes=None,  # fixed shapes only
    )

    model = onnx.load(str(output_path))
    onnx.checker.check_model(model)
    logger.info("ONNX exported and validated: %s", output_path)
    return output_path


def verify_onnx(probe: EmotionProbe, onnx_path: Path) -> float:
    """Verify ONNX output matches PyTorch. Returns max absolute difference."""
    import onnxruntime as ort

    dummy = torch.randn(*INPUT_SHAPE)
    with torch.no_grad():
        pt_out = probe(dummy).numpy()

    sess = ort.InferenceSession(str(onnx_path))
    ort_out = sess.run(None, {"latent": dummy.numpy()})[0]

    max_diff = float(np.abs(pt_out - ort_out).max())
    status = "PASS" if max_diff < 1e-5 else "FAIL"
    logger.info("ONNX verification: max_diff=%.2e  %s", max_diff, status)
    return max_diff


def print_model_summary(probe: EmotionProbe, onnx_path: Path) -> None:
    """Print parameter count and file size."""
    n_params = sum(p.numel() for p in probe.parameters())
    n_trainable = sum(p.numel() for p in probe.parameters() if p.requires_grad)
    file_size_kb = onnx_path.stat().st_size / 1024

    print("\n=== Emotion Probe Export Summary ===")
    print(f"  ONNX path:        {onnx_path}")
    print(f"  ONNX file size:   {file_size_kb:.1f} KB")
    print(f"  Parameters:       {n_params:,} total / {n_trainable:,} trainable")
    print(f"  Input shape:      {INPUT_SHAPE}  (latent)")
    print(f"  Output shape:     {OUTPUT_SHAPE}  (emotion: [valence, arousal])")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export trained EmotionProbe to ONNX for C++ plugin inference."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/emotion_probe/best_probe.pt"),
        help="Probe checkpoint path (default: checkpoints/emotion_probe/best_probe.pt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/emotion_probe_v01.onnx"),
        help="ONNX output path (default: models/emotion_probe_v01.onnx)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ckpt_path = (
        args.checkpoint if args.checkpoint.is_absolute() else Path(_PROJECT_ROOT) / args.checkpoint
    )
    out_path = (
        args.output if args.output.is_absolute() else Path(_PROJECT_ROOT) / args.output
    )

    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        return 1

    probe, ckpt = load_probe(ckpt_path)

    onnx_path = export_onnx(probe, out_path)

    max_diff = verify_onnx(probe, onnx_path)
    if max_diff >= 1e-5:
        logger.error("ONNX verification FAILED: max_diff=%.2e >= 1e-5", max_diff)
        return 1

    print_model_summary(probe, onnx_path)
    print(f"  Verification:     PASS (max_diff={max_diff:.2e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
