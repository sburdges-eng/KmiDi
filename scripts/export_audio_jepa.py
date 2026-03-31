#!/usr/bin/env python3
"""
Export Audio JEPA encoder to ONNX (and Core ML when supported) for Apple Silicon.

Usage:
    python scripts/export_audio_jepa.py
    python scripts/export_audio_jepa.py --checkpoint path/to/model.pt --output-dir models/
    python scripts/export_audio_jepa.py --benchmark --warmup 50 --iterations 200
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
import logging
import platform
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from music_brain.jepa.audio_jepa import AudioJEPAEncoder
from music_brain.jepa.config import AudioJEPAConfig

logger = logging.getLogger(__name__)

# Fixed input/output shapes for the medium-tier encoder
INPUT_SHAPE = (1, 1, 128, 512)  # (batch, channels, n_mels, max_frames)
OUTPUT_SHAPE = (1, 512, 256)    # (batch, time, latent_dim)


@dataclass
class ExportResult:
    onnx_path: Optional[Path] = None
    coreml_path: Optional[Path] = None
    onnx_verified: bool = False
    coreml_verified: bool = False


def load_encoder(checkpoint_path: str) -> AudioJEPAEncoder:
    """Load encoder from checkpoint with config reconstruction."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = AudioJEPAConfig(**ckpt["config"])
    encoder = AudioJEPAEncoder(config=config)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    logger.info(
        "Loaded encoder: tier=%s, latent_dim=%d, epoch=%d, loss=%.6f",
        config.tier, config.latent_dim, ckpt["epoch"], ckpt["loss"],
    )
    return encoder


def export_onnx(encoder: AudioJEPAEncoder, output_path: Path) -> Path:
    """Export encoder to fixed-shape ONNX (opset 17, batch=1)."""
    import onnx

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(*INPUT_SHAPE)

    torch.onnx.export(
        encoder,
        dummy,
        str(output_path),
        opset_version=18,
        input_names=["mel"],
        output_names=["latent"],
    )

    # Validate fixed shapes
    model = onnx.load(str(output_path))
    onnx.checker.check_model(model)
    inp_dims = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    assert inp_dims == list(INPUT_SHAPE), f"Dynamic shapes detected: {inp_dims}"

    logger.info("ONNX exported: %s", output_path)
    return output_path


def verify_onnx(encoder: AudioJEPAEncoder, onnx_path: Path) -> bool:
    """Verify ONNX output matches PyTorch within tolerance."""
    import onnxruntime as ort

    dummy = torch.randn(*INPUT_SHAPE)
    with torch.no_grad():
        pt_out = encoder(dummy).numpy()

    sess = ort.InferenceSession(str(onnx_path))
    ort_out = sess.run(None, {"mel": dummy.numpy()})[0]

    max_diff = np.abs(pt_out - ort_out).max()
    passed = max_diff < 1e-4
    logger.info("ONNX verification: max_diff=%.2e %s", max_diff, "PASS" if passed else "FAIL")
    return passed


def export_coreml(encoder: AudioJEPAEncoder, output_path: Path) -> Optional[Path]:
    """Export encoder to Core ML .mlpackage (ANE-preferred).

    NOTE: Requires coremltools with native bindings for the current Python version.
    As of 2026-03-31, coremltools 9.0 does not support Python 3.14.
    """
    if platform.system() != "Darwin":
        logger.warning("Core ML export requires macOS, skipping")
        return None

    try:
        import coremltools as ct
    except ImportError:
        logger.warning("coremltools not installed, skipping Core ML export")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(*INPUT_SHAPE)
    traced = torch.jit.trace(encoder, dummy)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="mel", shape=INPUT_SHAPE)],
        minimum_deployment_target=ct.target.macOS13,
        compute_units=ct.ComputeUnit.ALL,
    )
    mlmodel.author = "KmiDi"
    mlmodel.short_description = "Audio JEPA encoder - mel spectrogram to latent"
    mlmodel.version = "0.1"

    mlmodel.save(str(output_path))
    logger.info("Core ML exported: %s", output_path)
    return output_path


def verify_coreml(encoder: AudioJEPAEncoder, coreml_path: Path) -> bool:
    """Verify Core ML output matches PyTorch within tolerance."""
    if platform.system() != "Darwin":
        return False

    try:
        import coremltools as ct
    except ImportError:
        return False

    dummy = torch.randn(*INPUT_SHAPE)
    with torch.no_grad():
        pt_out = encoder(dummy).numpy()

    model = ct.models.MLModel(str(coreml_path))
    pred = model.predict({"mel": dummy.numpy()})
    cml_out = list(pred.values())[0]

    max_diff = np.abs(pt_out - cml_out).max()
    passed = max_diff < 1e-2
    logger.info("Core ML verification: max_diff=%.2e %s", max_diff, "PASS" if passed else "FAIL")
    return passed


def benchmark_onnx(onnx_path: Path, warmup: int = 50, iterations: int = 200) -> dict:
    """Benchmark ONNX Runtime inference latency."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path))
    dummy = np.random.randn(*INPUT_SHAPE).astype(np.float32)

    for _ in range(warmup):
        sess.run(None, {"mel": dummy})

    times = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        sess.run(None, {"mel": dummy})
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        times.append(elapsed_ms)

    times.sort()
    return {
        "runtime": "ONNX Runtime",
        "iterations": iterations,
        "p50_ms": round(np.percentile(times, 50), 3),
        "p95_ms": round(np.percentile(times, 95), 3),
        "p99_ms": round(np.percentile(times, 99), 3),
        "min_ms": round(min(times), 3),
        "max_ms": round(max(times), 3),
    }


def benchmark_coreml(coreml_path: Path, warmup: int = 50, iterations: int = 200) -> Optional[dict]:
    """Benchmark Core ML inference latency."""
    if platform.system() != "Darwin":
        return None

    try:
        import coremltools as ct
    except ImportError:
        return None

    model = ct.models.MLModel(str(coreml_path))
    dummy = np.random.randn(*INPUT_SHAPE).astype(np.float32)

    for _ in range(warmup):
        model.predict({"mel": dummy})

    times = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        model.predict({"mel": dummy})
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        times.append(elapsed_ms)

    times.sort()
    return {
        "runtime": "Core ML (ALL compute units)",
        "iterations": iterations,
        "p50_ms": round(np.percentile(times, 50), 3),
        "p95_ms": round(np.percentile(times, 95), 3),
        "p99_ms": round(np.percentile(times, 99), 3),
        "min_ms": round(min(times), 3),
        "max_ms": round(max(times), 3),
    }


def write_latency_report(
    output_dir: Path,
    onnx_bench: Optional[dict],
    coreml_bench: Optional[dict],
    result: ExportResult,
) -> Path:
    """Write bench/latency_report.md."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "latency_report.md"

    lines = [
        "# Audio JEPA Latency Report",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}",
        f"**Platform:** {platform.platform()}",
        f"**Input shape:** {INPUT_SHAPE} — 128 mels, 512 frames @ 22050 Hz",
        f"**Output shape:** {OUTPUT_SHAPE}",
        "",
        "## Artifacts",
        "",
        "| Format | Path | Verified |",
        "|--------|------|----------|",
        f"| ONNX | `{result.onnx_path}` | {'PASS' if result.onnx_verified else 'FAIL'} |",
        f"| Core ML | `{result.coreml_path or 'N/A (deferred)'}` | {'PASS' if result.coreml_verified else 'N/A'} |",
        "",
        "## Latency (warm-started, batch=1)",
        "",
        "| Runtime | p50 (ms) | p95 (ms) | p99 (ms) | min (ms) | max (ms) |",
        "|---------|----------|----------|----------|----------|----------|",
    ]

    for bench in [onnx_bench, coreml_bench]:
        if bench:
            lines.append(
                f"| {bench['runtime']} | {bench['p50_ms']} | {bench['p95_ms']} "
                f"| {bench['p99_ms']} | {bench['min_ms']} | {bench['max_ms']} |"
            )

    lines.append("")
    lines.append("## Target")
    lines.append("")
    lines.append("Phase 2 acceptance gate: forward latency <= 8 ms at batch=1 on Apple Silicon.")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Core ML export deferred: coremltools 9.0 lacks Python 3.14 native bindings.")
    lines.append("  Re-run with a Python 3.12 venv or when coremltools ships 3.14 support.")
    lines.append("")

    report_path.write_text("\n".join(lines))
    logger.info("Report written: %s", report_path)
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Export Audio JEPA to ONNX + Core ML")
    parser.add_argument(
        "--checkpoint", default="checkpoints/audio_jepa/best_model.pt",
        help="Path to .pt checkpoint",
    )
    parser.add_argument("--output-dir", default="models", help="Output directory for artifacts")
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmarks")
    parser.add_argument("--warmup", type=int, default=50, help="Benchmark warmup iterations")
    parser.add_argument("--iterations", type=int, default=200, help="Benchmark timed iterations")
    parser.add_argument("--coreml", action="store_true", help="Attempt Core ML export (requires compatible coremltools)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    output_dir = Path(args.output_dir)
    result = ExportResult()

    # Load
    encoder = load_encoder(args.checkpoint)

    # ONNX
    onnx_path = output_dir / "audio_jepa_v01.onnx"
    result.onnx_path = export_onnx(encoder, onnx_path)
    result.onnx_verified = verify_onnx(encoder, onnx_path)

    # Core ML (opt-in due to Python version constraints)
    if args.coreml:
        coreml_path = output_dir / "audio_jepa_v01.mlpackage"
        coreml_result = export_coreml(encoder, coreml_path)
        if coreml_result:
            result.coreml_path = coreml_result
            result.coreml_verified = verify_coreml(encoder, coreml_result)
    else:
        logger.info("Core ML export skipped (use --coreml to attempt)")

    # Benchmark
    onnx_bench = None
    coreml_bench = None
    if args.benchmark:
        logger.info("Benchmarking (%d warmup, %d iterations)...", args.warmup, args.iterations)
        onnx_bench = benchmark_onnx(onnx_path, args.warmup, args.iterations)
        logger.info("ONNX: p50=%.1fms p95=%.1fms p99=%.1fms",
                     onnx_bench["p50_ms"], onnx_bench["p95_ms"], onnx_bench["p99_ms"])

        if result.coreml_path:
            coreml_bench = benchmark_coreml(result.coreml_path, args.warmup, args.iterations)
            if coreml_bench:
                logger.info("Core ML: p50=%.1fms p95=%.1fms p99=%.1fms",
                             coreml_bench["p50_ms"], coreml_bench["p95_ms"], coreml_bench["p99_ms"])

    # Report
    if args.benchmark:
        write_latency_report(Path("bench"), onnx_bench, coreml_bench, result)

    # Summary
    print("\n=== Export Summary ===")
    print(f"ONNX:    {result.onnx_path} (verified={result.onnx_verified})")
    print(f"CoreML:  {result.coreml_path or 'N/A (deferred)'} (verified={result.coreml_verified})")
    if onnx_bench:
        print(f"ONNX latency:    p50={onnx_bench['p50_ms']}ms p99={onnx_bench['p99_ms']}ms")
    if coreml_bench:
        print(f"CoreML latency:  p50={coreml_bench['p50_ms']}ms p99={coreml_bench['p99_ms']}ms")


if __name__ == "__main__":
    main()
