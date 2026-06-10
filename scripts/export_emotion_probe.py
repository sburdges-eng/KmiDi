#!/usr/bin/env python3
"""
Export trained EmotionProbe to ONNX and Core ML for use in the C++ plugin.

Loads the best probe checkpoint (from train_emotion_probe.py), traces with
a dummy latent vector, and exports to ONNX and Core ML with fixed input/output
shapes. Verifies numerical parity and prints a model summary.

Usage:
    python scripts/export_emotion_probe.py
    python scripts/export_emotion_probe.py \
        --checkpoint checkpoints/emotion_probe/best_probe.pt \
        --output-dir models/
    python scripts/export_emotion_probe.py --benchmark
"""

from __future__ import annotations

import argparse
import logging
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from music_brain.jepa.emotion_probe import EmotionProbe

logger = logging.getLogger(__name__)

# Ensure project root is on sys.path when running as a script
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Fixed I/O shapes (batch=1 for C++ plugin)
INPUT_SHAPE = (1, 256)  # (batch, latent_dim)
OUTPUT_SHAPE = (1, 2)  # (batch, [valence, arousal])


@dataclass
class ExportResult:
    onnx_path: Optional[Path] = None
    coreml_path: Optional[Path] = None
    onnx_verified: bool = False
    coreml_verified: bool = False


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
        ckpt.get("epoch", -1),
        ckpt.get("val_loss", float("nan")),
        latent_dim,
        hidden_dim,
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


def verify_onnx(probe: EmotionProbe, onnx_path: Path) -> bool:
    """Verify ONNX output matches PyTorch."""
    import onnxruntime as ort

    dummy = torch.randn(*INPUT_SHAPE)
    with torch.no_grad():
        pt_out = probe(dummy).numpy()

    sess = ort.InferenceSession(str(onnx_path))
    ort_out = sess.run(None, {"latent": dummy.numpy()})[0]

    max_diff = float(np.abs(pt_out - ort_out).max())
    passed = max_diff < 1e-5
    logger.info("ONNX verification: max_diff=%.2e  %s", max_diff, "PASS" if passed else "FAIL")
    return passed


def export_coreml(probe: EmotionProbe, output_path: Path) -> Optional[Path]:
    """Export probe to Core ML .mlpackage."""
    if platform.system() != "Darwin":
        logger.warning("Core ML export requires macOS, skipping")
        return None

    import coremltools as ct

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(*INPUT_SHAPE)
    traced = torch.jit.trace(probe, dummy)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="latent", shape=INPUT_SHAPE)],
        minimum_deployment_target=ct.target.macOS13,
        compute_units=ct.ComputeUnit.ALL,
    )
    mlmodel.author = "KmiDi"
    mlmodel.short_description = "Emotion Probe MLP - JEPA latent to valence/arousal"
    mlmodel.version = "0.1"

    mlmodel.save(str(output_path))
    logger.info("Core ML exported: %s", output_path)
    return output_path


def verify_coreml(probe: EmotionProbe, coreml_path: Path) -> bool:
    """Verify Core ML output matches PyTorch."""
    if platform.system() != "Darwin":
        return False

    import coremltools as ct

    dummy = torch.randn(*INPUT_SHAPE)
    with torch.no_grad():
        pt_out = probe(dummy).numpy()

    model = ct.models.MLModel(str(coreml_path))
    pred = model.predict({"latent": dummy.numpy()})
    cml_out = list(pred.values())[0]

    max_diff = float(np.abs(pt_out - cml_out).max())
    passed = max_diff < 1e-3
    logger.info("Core ML verification: max_diff=%.2e  %s", max_diff, "PASS" if passed else "FAIL")
    return passed


def benchmark_model(model_path: Path, is_coreml: bool = False, iterations: int = 200) -> dict:
    """Benchmark inference latency."""
    dummy = np.random.randn(*INPUT_SHAPE).astype(np.float32)
    times = []

    if is_coreml:
        import coremltools as ct

        model = ct.models.MLModel(str(model_path))
        # Warmup
        for _ in range(20):
            model.predict({"latent": dummy})
        # Timed
        for _ in range(iterations):
            start = time.perf_counter_ns()
            model.predict({"latent": dummy})
            times.append((time.perf_counter_ns() - start) / 1e6)
        runtime_name = "Core ML"
    else:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(model_path))
        # Warmup
        for _ in range(20):
            sess.run(None, {"latent": dummy})
        # Timed
        for _ in range(iterations):
            start = time.perf_counter_ns()
            sess.run(None, {"latent": dummy})
            times.append((time.perf_counter_ns() - start) / 1e6)
        runtime_name = "ONNX Runtime"

    times.sort()
    return {
        "runtime": runtime_name,
        "p50_ms": round(np.percentile(times, 50), 3),
        "p99_ms": round(np.percentile(times, 99), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export trained EmotionProbe to ONNX and Core ML.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/emotion_probe/best_probe.pt"),
        help="Probe checkpoint path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Output directory for artifacts",
    )
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmarks")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    _PR = Path(_PROJECT_ROOT)
    ckpt_path = args.checkpoint if args.checkpoint.is_absolute() else _PR / args.checkpoint
    output_dir = args.output_dir if args.output_dir.is_absolute() else _PR / args.output_dir

    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        return 1

    probe, ckpt = load_probe(ckpt_path)
    result = ExportResult()

    # ONNX
    onnx_path = output_dir / "emotion_probe_v01.onnx"
    result.onnx_path = export_onnx(probe, onnx_path)
    result.onnx_verified = verify_onnx(probe, onnx_path)

    # Core ML
    if platform.system() == "Darwin":
        coreml_path = output_dir / "emotion_probe_v01.mlpackage"
        result.coreml_path = export_coreml(probe, coreml_path)
        if result.coreml_path:
            result.coreml_verified = verify_coreml(probe, result.coreml_path)

    # Summary
    print("\n=== Emotion Probe Export Summary ===")
    print(f"  ONNX:     {result.onnx_path} (verified={result.onnx_verified})")
    print(f"  CoreML:   {result.coreml_path or 'N/A'} (verified={result.coreml_verified})")

    if args.benchmark:
        print("\n  Latency (p50 / p99):")
        if result.onnx_path:
            b = benchmark_model(result.onnx_path)
            print(f"    {b['runtime']}: {b['p50_ms']}ms / {b['p99_ms']}ms")
        if result.coreml_path:
            b = benchmark_model(result.coreml_path, is_coreml=True)
            print(f"    {b['runtime']}:   {b['p50_ms']}ms / {b['p99_ms']}ms")

    return 0


if __name__ == "__main__":
    sys.exit(main())
