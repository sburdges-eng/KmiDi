#!/usr/bin/env python3
"""Export trained emotion AudioClassifier to ONNX and Core ML.

Loads a ``best.pt`` written by ``training/scripts/train_emotion_local.py``
(CNN over mel spectrograms → class logits), traces with a dummy mel batch,
and exports to ONNX (dynamic time axis) and Core ML (.mlpackage). Verifies
numerical parity between PyTorch and ONNX Runtime.

This is a *mel-to-class* emotion model. It is NOT a drop-in for
``AudioEmotionRunner``'s probe slot (that path consumes JEPA latents →
valence/arousal, see ``export_emotion_probe.py``).

Usage::

    python scripts/export_emotion_classifier.py \\
        --checkpoint ~/Models/checkpoints/emotion-local-combined-50ep-*/best.pt

    python scripts/export_emotion_classifier.py \\
        --checkpoint <path> --output-dir models/ --benchmark
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "training"))

from src.models.audio_classifier import AudioClassifier  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("export_emotion_classifier")

# Fixed input shape for Core ML (time is variable on ONNX).
#   [batch=1, channel=1, n_mels, time]
DEFAULT_N_MELS = 64
DEFAULT_TIME = 96   # ~3.0s @ 16kHz with hop=512 → 94 frames, round up


@dataclass
class ExportResult:
    onnx_path: Optional[Path] = None
    coreml_path: Optional[Path] = None
    onnx_verified: bool = False
    coreml_verified: bool = False
    num_classes: int = 0
    class_names: list[str] | None = None


def load_classifier(checkpoint_path: Path) -> tuple[AudioClassifier, dict]:
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    num_classes = int(ckpt["num_classes"])
    # n_mels isn't serialized in the ckpt; default matches trainer default.
    state_dict = ckpt["model_state_dict"]
    model = AudioClassifier(num_classes=num_classes, n_mels=DEFAULT_N_MELS)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(
        "Loaded classifier: epoch=%d  val_acc=%.4f  num_classes=%d  classes=%s",
        ckpt.get("epoch", -1),
        ckpt.get("val_acc", float("nan")),
        num_classes,
        ckpt.get("class_names"),
    )
    return model, ckpt


def export_onnx(model: AudioClassifier, output_path: Path, n_mels: int) -> Path:
    import onnx

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 1, n_mels, DEFAULT_TIME)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=17,
        input_names=["mel"],
        output_names=["logits"],
        # Time is variable; batch left at 1 for plugin use.
        dynamic_axes={"mel": {3: "time"}, "logits": {}},
    )
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX exported and validated: %s", output_path)
    return output_path


def verify_onnx(model: AudioClassifier, onnx_path: Path, n_mels: int,
                trials: int = 4) -> tuple[bool, float]:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path))
    rng = np.random.default_rng(42)
    max_diff = 0.0
    for _ in range(trials):
        # Exercise variable time axis.
        t = int(rng.integers(48, 160))
        mel = rng.standard_normal((1, 1, n_mels, t)).astype(np.float32)
        with torch.no_grad():
            pt_out = model(torch.from_numpy(mel)).numpy()
        ort_out = sess.run(None, {"mel": mel})[0]
        max_diff = max(max_diff, float(np.abs(pt_out - ort_out).max()))

    passed = max_diff < 1e-4
    logger.info("ONNX verification across %d trials: max_diff=%.2e  %s",
                trials, max_diff, "PASS" if passed else "FAIL")
    return passed, max_diff


def export_coreml(model: AudioClassifier, output_path: Path, n_mels: int) -> Optional[Path]:
    if platform.system() != "Darwin":
        logger.warning("Core ML export requires macOS — skipping.")
        return None
    import coremltools as ct

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Core ML doesn't like TorchScript traces with dynamic dims from our CNN
    # (AdaptiveAvgPool2d handles variable time just fine at runtime, but the
    # mlprogram converter is easier to drive with a fixed-time trace).
    dummy = torch.randn(1, 1, n_mels, DEFAULT_TIME)
    traced = torch.jit.trace(model, dummy)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="mel", shape=dummy.shape)],
        outputs=[ct.TensorType(name="logits")],
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL,   # ANE-preferred
        minimum_deployment_target=ct.target.macOS14,
    )
    mlmodel.save(str(output_path))
    logger.info("Core ML exported: %s", output_path)
    return output_path


def verify_coreml(model: AudioClassifier, coreml_path: Path, n_mels: int) -> tuple[bool, float]:
    if platform.system() != "Darwin":
        return False, float("nan")
    import coremltools as ct

    mlmodel = ct.models.MLModel(str(coreml_path))
    mel = np.random.default_rng(0).standard_normal((1, 1, n_mels, DEFAULT_TIME)).astype(np.float32)
    with torch.no_grad():
        pt_out = model(torch.from_numpy(mel)).numpy()
    cm_out = mlmodel.predict({"mel": mel})
    # Output key is the traced name; be robust about it.
    cm_arr = cm_out.get("logits") or next(iter(cm_out.values()))
    max_diff = float(np.abs(pt_out - np.asarray(cm_arr)).max())
    # Core ML's CPU/GPU/ANE fused kernels diverge more than ORT; relax threshold.
    passed = max_diff < 5e-3
    logger.info("Core ML verification: max_diff=%.2e  %s",
                max_diff, "PASS" if passed else "FAIL")
    return passed, max_diff


def benchmark(model: AudioClassifier, onnx_path: Path, coreml_path: Optional[Path],
              n_mels: int, runs: int = 200) -> None:
    import onnxruntime as ort

    mel = np.random.default_rng(1).standard_normal((1, 1, n_mels, DEFAULT_TIME)).astype(np.float32)

    # PyTorch CPU
    pt_tensor = torch.from_numpy(mel)
    with torch.no_grad():
        for _ in range(10):
            _ = model(pt_tensor)
        t0 = time.perf_counter()
        for _ in range(runs):
            _ = model(pt_tensor)
        dt_pt = (time.perf_counter() - t0) / runs * 1000

    # ONNX Runtime
    sess = ort.InferenceSession(str(onnx_path))
    for _ in range(10):
        sess.run(None, {"mel": mel})
    t0 = time.perf_counter()
    for _ in range(runs):
        sess.run(None, {"mel": mel})
    dt_ort = (time.perf_counter() - t0) / runs * 1000

    logger.info("Bench (ms/inference @ time=%d, %d runs): torch=%.3f  ort=%.3f",
                DEFAULT_TIME, runs, dt_pt, dt_ort)

    if coreml_path and platform.system() == "Darwin":
        import coremltools as ct
        ml = ct.models.MLModel(str(coreml_path))
        for _ in range(10):
            ml.predict({"mel": mel})
        t0 = time.perf_counter()
        for _ in range(runs):
            ml.predict({"mel": mel})
        dt_cm = (time.perf_counter() - t0) / runs * 1000
        logger.info("Bench Core ML: %.3f ms/inference", dt_cm)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="Path to best.pt from train_emotion_local.py")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="Directory for .onnx and .mlpackage. "
                         "Default: <checkpoint-dir>/export/")
    ap.add_argument("--n-mels", type=int, default=DEFAULT_N_MELS)
    ap.add_argument("--skip-coreml", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        return 2

    out_dir = Path(args.output_dir).expanduser() if args.output_dir \
        else ckpt_path.parent / "export"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, ckpt = load_classifier(ckpt_path)
    class_names = ckpt.get("class_names") or []

    result = ExportResult(num_classes=int(ckpt["num_classes"]), class_names=class_names)

    # ONNX
    onnx_path = out_dir / "emotion_classifier.onnx"
    result.onnx_path = export_onnx(model, onnx_path, args.n_mels)
    result.onnx_verified, _ = verify_onnx(model, onnx_path, args.n_mels)

    # Core ML
    if not args.skip_coreml:
        coreml_path = out_dir / "emotion_classifier.mlpackage"
        result.coreml_path = export_coreml(model, coreml_path, args.n_mels)
        if result.coreml_path is not None:
            result.coreml_verified, _ = verify_coreml(model, coreml_path, args.n_mels)

    # Sidecar metadata (class labels + input contract) for consumers.
    (out_dir / "emotion_classifier.meta.json").write_text(json.dumps({
        "source_checkpoint": str(ckpt_path),
        "source_epoch": int(ckpt.get("epoch", -1)),
        "source_val_acc": float(ckpt.get("val_acc", float("nan"))),
        "num_classes": result.num_classes,
        "class_names": class_names,
        "input": {
            "name": "mel",
            "layout": "NCHW",
            "shape": [1, 1, args.n_mels, "time (variable on ONNX)"],
            "sample_rate_hint": 16000,
            "n_mels": args.n_mels,
        },
        "output": {"name": "logits", "shape": [1, result.num_classes]},
    }, indent=2))

    if args.benchmark:
        benchmark(model, result.onnx_path, result.coreml_path, args.n_mels)

    logger.info("Export summary: ONNX=%s  verified=%s  CoreML=%s  verified=%s",
                result.onnx_path, result.onnx_verified,
                result.coreml_path, result.coreml_verified)
    return 0 if result.onnx_verified else 1


if __name__ == "__main__":
    sys.exit(main())
