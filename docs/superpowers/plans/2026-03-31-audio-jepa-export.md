# Audio JEPA Export & Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export the trained Audio JEPA encoder to fixed-shape ONNX and ANE-optimized Core ML, verify both artifacts, and benchmark inference latency on Apple Silicon.

**Architecture:** Load the checkpoint's encoder state dict into `AudioJEPAEncoder`, export to ONNX (opset 17, fully fixed shapes), convert ONNX to Core ML `.mlpackage` via `coremltools`, then run warm-started latency benchmarks measuring p50/p95/p99. All artifacts are local and offline.

**Tech Stack:** PyTorch 2.11, onnx 1.20, onnxruntime, coremltools, Apple Silicon MPS/ANE

---

## File Map

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `scripts/export_audio_jepa.py` | CLI entry point: load checkpoint, export ONNX + Core ML, verify, benchmark, write report |
| Create | `tests/unit/test_export_audio_jepa.py` | Tests for checkpoint loading, ONNX export, Core ML export, shape verification |
| Create (by script) | `models/audio_jepa_v01.onnx` | Exported ONNX artifact (gitignored) |
| Create (by script) | `models/audio_jepa_v01.mlpackage/` | Exported Core ML artifact (gitignored) |
| Create (by script) | `bench/latency_report.md` | Benchmark results |
| Modify | `.gitignore` | Ensure `models/*.onnx`, `models/*.mlpackage`, `bench/` are gitignored |

---

### Task 1: Install missing dependencies

**Files:**
- Modify: `requirements.txt` or install directly into venv

- [ ] **Step 1: Install onnxruntime and coremltools into the KmiDi venv**

```bash
cd /Users/seanburdges/Dev/KmiDi
source venv/bin/activate
pip install onnxruntime coremltools
```

- [ ] **Step 2: Verify installations**

```bash
source venv/bin/activate
python3 -c "import onnxruntime; print('ort:', onnxruntime.__version__)"
python3 -c "import coremltools; print('ct:', coremltools.__version__)"
```

Expected: Both print version numbers without error.

---

### Task 2: Update .gitignore for export artifacts

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Check current .gitignore for models/ and bench/ patterns**

```bash
grep -n 'models/' .gitignore; grep -n 'bench/' .gitignore
```

- [ ] **Step 2: Add gitignore entries if missing**

Append to `.gitignore`:

```
# Export artifacts (regenerable from checkpoint)
models/*.onnx
models/*.mlpackage/
bench/
```

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: gitignore ONNX/Core ML export artifacts and bench/"
```

---

### Task 3: Write tests for checkpoint loading and ONNX export

**Files:**
- Create: `tests/unit/test_export_audio_jepa.py`

- [ ] **Step 1: Write failing tests for checkpoint loading and ONNX export**

```python
"""Tests for Audio JEPA export pipeline."""

import os
import tempfile

import pytest
import torch

from music_brain.jepa.audio_jepa import AudioJEPAEncoder
from music_brain.jepa.config import AudioJEPAConfig


CHECKPOINT_PATH = "checkpoints/audio_jepa/best_model.pt"


@pytest.fixture
def checkpoint():
    """Load the real checkpoint."""
    assert os.path.exists(CHECKPOINT_PATH), f"Checkpoint not found: {CHECKPOINT_PATH}"
    return torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)


@pytest.fixture
def encoder(checkpoint):
    """Reconstruct encoder from checkpoint config and weights."""
    config = AudioJEPAConfig(**checkpoint["config"])
    enc = AudioJEPAEncoder(config=config)
    enc.load_state_dict(checkpoint["encoder"])
    enc.eval()
    return enc


class TestCheckpointLoading:
    def test_checkpoint_has_required_keys(self, checkpoint):
        for key in ("encoder", "config", "epoch", "loss"):
            assert key in checkpoint, f"Missing key: {key}"

    def test_config_matches_expected_shape(self, checkpoint):
        cfg = checkpoint["config"]
        assert cfg["n_mels"] == 128
        assert cfg["max_frames"] == 512
        assert cfg["latent_dim"] == 256
        assert cfg["tier"] == "medium"

    def test_encoder_loads_and_runs(self, encoder):
        x = torch.randn(1, 1, 128, 512)
        with torch.no_grad():
            z = encoder(x)
        assert z.shape == (1, 512, 256)


class TestOnnxExport:
    def test_onnx_export_produces_valid_file(self, encoder):
        import onnx

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test.onnx")
            dummy = torch.randn(1, 1, 128, 512)
            torch.onnx.export(
                encoder,
                dummy,
                out_path,
                opset_version=17,
                input_names=["mel"],
                output_names=["latent"],
            )
            assert os.path.exists(out_path)
            model = onnx.load(out_path)
            onnx.checker.check_model(model)

    def test_onnx_has_fixed_shapes(self, encoder):
        import onnx

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test.onnx")
            dummy = torch.randn(1, 1, 128, 512)
            torch.onnx.export(
                encoder,
                dummy,
                out_path,
                opset_version=17,
                input_names=["mel"],
                output_names=["latent"],
            )
            model = onnx.load(out_path)
            inp = model.graph.input[0]
            dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            assert dims == [1, 1, 128, 512], f"Expected fixed [1,1,128,512], got {dims}"

    def test_onnx_inference_matches_pytorch(self, encoder):
        import onnxruntime as ort
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test.onnx")
            dummy = torch.randn(1, 1, 128, 512)
            torch.onnx.export(
                encoder,
                dummy,
                out_path,
                opset_version=17,
                input_names=["mel"],
                output_names=["latent"],
            )
            # PyTorch reference
            with torch.no_grad():
                pt_out = encoder(dummy).numpy()

            # ONNX Runtime
            sess = ort.InferenceSession(out_path)
            ort_out = sess.run(None, {"mel": dummy.numpy()})[0]

            np.testing.assert_allclose(pt_out, ort_out, atol=1e-5, rtol=1e-4)
```

- [ ] **Step 2: Run tests to verify they fail (encoder fixture works but export script doesn't exist yet)**

```bash
source venv/bin/activate
python3 -m pytest tests/unit/test_export_audio_jepa.py -v
```

Expected: All tests in `TestCheckpointLoading` PASS (they use existing code). `TestOnnxExport` tests PASS too (they inline the export logic). This is expected — these tests validate the export approach before we wrap it in a script.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_export_audio_jepa.py
git commit -m "test: add Audio JEPA export pipeline tests"
```

---

### Task 4: Write tests for Core ML export

**Files:**
- Modify: `tests/unit/test_export_audio_jepa.py`

- [ ] **Step 1: Add Core ML export tests**

Append to `tests/unit/test_export_audio_jepa.py`:

```python
class TestCoremlExport:
    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="Core ML tests require macOS with Apple Silicon",
    )
    def test_coreml_export_produces_mlpackage(self, encoder):
        import coremltools as ct

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test.mlpackage")
            dummy = torch.randn(1, 1, 128, 512)
            traced = torch.jit.trace(encoder, dummy)
            mlmodel = ct.convert(
                traced,
                inputs=[ct.TensorType(name="mel", shape=(1, 1, 128, 512))],
                minimum_deployment_target=ct.target.macOS13,
                compute_units=ct.ComputeUnit.ALL,
            )
            mlmodel.save(out_path)
            assert os.path.exists(out_path)

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="Core ML tests require macOS with Apple Silicon",
    )
    def test_coreml_inference_matches_pytorch(self, encoder):
        import coremltools as ct
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test.mlpackage")
            dummy = torch.randn(1, 1, 128, 512)
            traced = torch.jit.trace(encoder, dummy)
            mlmodel = ct.convert(
                traced,
                inputs=[ct.TensorType(name="mel", shape=(1, 1, 128, 512))],
                minimum_deployment_target=ct.target.macOS13,
                compute_units=ct.ComputeUnit.ALL,
            )
            mlmodel.save(out_path)
            loaded = ct.models.MLModel(out_path)

            # PyTorch reference
            with torch.no_grad():
                pt_out = encoder(dummy).numpy()

            # Core ML
            pred = loaded.predict({"mel": dummy.numpy()})
            cml_out = list(pred.values())[0]

            np.testing.assert_allclose(pt_out, cml_out, atol=1e-3, rtol=1e-3)
```

- [ ] **Step 2: Run Core ML tests**

```bash
source venv/bin/activate
python3 -m pytest tests/unit/test_export_audio_jepa.py::TestCoremlExport -v
```

Expected: PASS on macOS with Apple Silicon.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_export_audio_jepa.py
git commit -m "test: add Core ML export and parity tests"
```

---

### Task 5: Write the export script

**Files:**
- Create: `scripts/export_audio_jepa.py`

- [ ] **Step 1: Create the export script**

```python
#!/usr/bin/env python3
"""
Export Audio JEPA encoder to ONNX and Core ML for Apple Silicon deployment.

Usage:
    python scripts/export_audio_jepa.py
    python scripts/export_audio_jepa.py --checkpoint path/to/model.pt --output-dir models/
    python scripts/export_audio_jepa.py --benchmark --warmup 50 --iterations 200
"""

from __future__ import annotations

import argparse
import logging
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from music_brain.jepa.audio_jepa import AudioJEPAEncoder
from music_brain.jepa.config import AudioJEPAConfig

logger = logging.getLogger(__name__)


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
    dummy = torch.randn(1, 1, 128, 512)

    torch.onnx.export(
        encoder,
        dummy,
        str(output_path),
        opset_version=17,
        input_names=["mel"],
        output_names=["latent"],
    )

    # Validate fixed shapes
    model = onnx.load(str(output_path))
    onnx.checker.check_model(model)
    inp_dims = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    assert inp_dims == [1, 1, 128, 512], f"Dynamic shapes detected: {inp_dims}"

    logger.info("ONNX exported: %s", output_path)
    return output_path


def verify_onnx(encoder: AudioJEPAEncoder, onnx_path: Path) -> bool:
    """Verify ONNX output matches PyTorch within tolerance."""
    import onnxruntime as ort

    dummy = torch.randn(1, 1, 128, 512)
    with torch.no_grad():
        pt_out = encoder(dummy).numpy()

    sess = ort.InferenceSession(str(onnx_path))
    ort_out = sess.run(None, {"mel": dummy.numpy()})[0]

    max_diff = np.abs(pt_out - ort_out).max()
    passed = max_diff < 1e-4
    logger.info("ONNX verification: max_diff=%.2e %s", max_diff, "PASS" if passed else "FAIL")
    return passed


def export_coreml(encoder: AudioJEPAEncoder, output_path: Path) -> Optional[Path]:
    """Export encoder to Core ML .mlpackage (ANE-preferred)."""
    if platform.system() != "Darwin":
        logger.warning("Core ML export requires macOS, skipping")
        return None

    import coremltools as ct

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 1, 128, 512)
    traced = torch.jit.trace(encoder, dummy)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="mel", shape=(1, 1, 128, 512))],
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

    import coremltools as ct

    dummy = torch.randn(1, 1, 128, 512)
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
    dummy = np.random.randn(1, 1, 128, 512).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        sess.run(None, {"mel": dummy})

    # Timed runs
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

    import coremltools as ct

    model = ct.models.MLModel(str(coreml_path))
    dummy = np.random.randn(1, 1, 128, 512).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        model.predict({"mel": dummy})

    # Timed runs
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
        f"**Input shape:** (1, 1, 128, 512) — 128 mels, 512 frames @ 22050 Hz",
        f"**Output shape:** (1, 512, 256)",
        "",
        "## Artifacts",
        "",
        f"| Format | Path | Verified |",
        f"|--------|------|----------|",
        f"| ONNX | `{result.onnx_path}` | {'PASS' if result.onnx_verified else 'FAIL'} |",
        f"| Core ML | `{result.coreml_path}` | {'PASS' if result.coreml_verified else 'FAIL/N/A'} |",
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
    parser.add_argument("--skip-coreml", action="store_true", help="Skip Core ML export")
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

    # Core ML
    if not args.skip_coreml:
        coreml_path = output_dir / "audio_jepa_v01.mlpackage"
        coreml_result = export_coreml(encoder, coreml_path)
        if coreml_result:
            result.coreml_path = coreml_result
            result.coreml_verified = verify_coreml(encoder, coreml_result)

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
    print(f"CoreML:  {result.coreml_path} (verified={result.coreml_verified})")
    if onnx_bench:
        print(f"ONNX latency:    p50={onnx_bench['p50_ms']}ms p99={onnx_bench['p99_ms']}ms")
    if coreml_bench:
        print(f"CoreML latency:  p50={coreml_bench['p50_ms']}ms p99={coreml_bench['p99_ms']}ms")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script runs (export only, no benchmark)**

```bash
source venv/bin/activate
python3 scripts/export_audio_jepa.py --output-dir models/
```

Expected: Creates `models/audio_jepa_v01.onnx` and `models/audio_jepa_v01.mlpackage/`, prints verification results.

- [ ] **Step 3: Commit**

```bash
git add scripts/export_audio_jepa.py
git commit -m "feat: add Audio JEPA export script (ONNX + Core ML + benchmark)"
```

---

### Task 6: Run full export with benchmarks

**Files:**
- Creates: `models/audio_jepa_v01.onnx`, `models/audio_jepa_v01.mlpackage/`, `bench/latency_report.md`

- [ ] **Step 1: Run export with benchmarks**

```bash
source venv/bin/activate
python3 scripts/export_audio_jepa.py --benchmark --warmup 50 --iterations 200
```

Expected: ONNX and Core ML both export and verify. Benchmark results printed. `bench/latency_report.md` created.

- [ ] **Step 2: Review latency report**

```bash
cat bench/latency_report.md
```

Check: Are p50/p95/p99 within the 5-8ms target? If not, note it — the numbers are the numbers. The report is the deliverable regardless.

- [ ] **Step 3: Run the full test suite to confirm nothing broke**

```bash
source venv/bin/activate
python3 -m pytest tests/unit/test_export_audio_jepa.py tests/unit/test_jepa_models.py -v
```

Expected: All tests PASS.

- [ ] **Step 4: Commit the latency report (not the model artifacts)**

```bash
git add bench/latency_report.md
git commit -m "bench: Audio JEPA latency report — ONNX + Core ML on Apple Silicon"
```

---

### Task 7: Final verification

- [ ] **Step 1: Verify exported ONNX file size is reasonable**

```bash
ls -lh models/audio_jepa_v01.onnx
```

Expected: ~33MB (similar to checkpoint since it's the encoder weights only).

- [ ] **Step 2: Verify ONNX has no dynamic axes**

```bash
source venv/bin/activate
python3 -c "
import onnx
m = onnx.load('models/audio_jepa_v01.onnx')
for inp in m.graph.input:
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f'{inp.name}: {dims}')
for out in m.graph.output:
    dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
    print(f'{out.name}: {dims}')
"
```

Expected:
```
mel: [1, 1, 128, 512]
latent: [1, 512, 256]
```

- [ ] **Step 3: Verify Core ML can predict**

```bash
source venv/bin/activate
python3 -c "
import coremltools as ct
import numpy as np
m = ct.models.MLModel('models/audio_jepa_v01.mlpackage')
out = m.predict({'mel': np.random.randn(1,1,128,512).astype(np.float32)})
print('Output shape:', list(out.values())[0].shape)
"
```

Expected: `Output shape: (1, 512, 256)`

- [ ] **Step 4: Final commit if any adjustments were made**

```bash
git add -A
git status
# Only commit if there are changes
git commit -m "chore: finalize Audio JEPA export pipeline"
```
