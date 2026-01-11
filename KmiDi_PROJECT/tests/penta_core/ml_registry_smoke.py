"""
Smoke test for MLInterface registry loading and stub inference.

Assumptions:
- The pybind module `penta_core_native` has been built and is on PYTHONPATH
  (e.g., `${build_dir}/python` or installed into the venv).
- models/registry.json exists at repo root and references stub JSON models.

This test exercises:
1) Loading registry.json via MLInterface.load_registry
2) Starting/stopping the inference thread
3) Submitting a feature vector and polling for a result (stub echoes input)
"""

from __future__ import annotations

import sys
import time
from array import array
from pathlib import Path


def run_smoke() -> None:
    # Resolve repo root from test location: tests/penta_core/ -> repo root is parents[2]
    root = Path(__file__).resolve().parents[2]
    models_dir = root / "models"
    registry_path = models_dir / "registry.json"

    if not registry_path.exists():
        raise FileNotFoundError(f"registry.json not found at {registry_path}")

    # Import pybind module
    try:
        import penta_core_native as pcn
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "penta_core_native not importable. "
            "Build the bindings (cmake --build ... --target penta_core_native) "
            "and ensure the output directory is on PYTHONPATH."
        ) from exc

    ml = pcn.ml

    # Configure interface
    config = ml.MLConfig()
    config.model_directory = str(models_dir)

    iface = ml.MLInterface(config)

    # Load registry
    loaded = iface.load_registry(str(registry_path))
    assert loaded, "Failed to load registry.json"

    # Start inference thread
    assert iface.start(), "Failed to start MLInterface thread"

    try:
        # Prepare a small feature vector for EmotionRecognizer (input 128 floats; use 64 to keep short)
        features = array("f", [0.1] * 64)
        queued, req_id = iface.submit_features(ml.ModelType.EmotionRecognizer, features, timestamp=0)
        assert queued, "Request queue rejected submission"

        # Poll for result (stub should echo input)
        result = None
        deadline = time.time() + 2.0
        while time.time() < deadline:
            polled = iface.poll_result()
            if polled is not None:
                result = polled
                break
            time.sleep(0.01)

        assert result is not None, "No inference result received"
        assert result["request_id"] == req_id, "Mismatched request_id"
        assert result["success"], "Inference reported failure"
        assert result["output_size"] == len(features), "Unexpected output size"

    finally:
        iface.stop()


if __name__ == "__main__":
    run_smoke()

