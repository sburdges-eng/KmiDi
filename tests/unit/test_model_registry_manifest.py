from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("module_path", "module_name"),
    [
        (REPO_ROOT / "python" / "penta_core" / "ml" / "model_registry.py", "root_model_registry"),
        (REPO_ROOT / "music_brain" / "penta_core" / "ml" / "model_registry.py", "music_brain_model_registry"),
    ],
)
def test_manifest_loader_supports_jepa_and_language_model_entries(tmp_path, module_path, module_name):
    module = load_module(module_path, module_name)

    for filename in ("audiojepa.onnx", "chordjepa.onnx", "llama.onnx"):
        (tmp_path / filename).write_bytes(b"stub")

    manifest = {
        "registry_version": "1.1",
        "models": [
            {
                "id": "audiojepa",
                "file": "audiojepa.onnx",
                "format": "onnx",
                "task": "audio_jepa",
                "input_size": 256,
                "output_size": 128,
                "status": "stub",
                "exports": {"onnx_path": "audiojepa.onnx"},
            },
            {
                "id": "chordjepa",
                "file": "chordjepa.onnx",
                "format": "onnx",
                "task": "chord_jepa",
                "input_size": 128,
                "output_size": 64,
                "status": "stub",
                "exports": {"onnx_path": "chordjepa.onnx"},
            },
            {
                "id": "languagemodel",
                "file": "llama_onnx.json",
                "format": "onnx",
                "task": "language_model",
                "input_size": 512,
                "output_size": 128,
                "status": "stub",
                "exports": {"onnx_path": "llama.onnx"},
            },
        ],
    }
    manifest_path = tmp_path / "registry.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    registry = module.ModelRegistry()
    registry._models = {}

    loaded = registry.load_registry_manifest(str(manifest_path), validate=False)

    assert loaded == 3
    assert registry.get("audiojepa").task == module.ModelTask.AUDIO_JEPA
    assert registry.get("chordjepa").task == module.ModelTask.CHORD_JEPA
    assert registry.get("languagemodel").task == module.ModelTask.LANGUAGE_MODEL
    assert registry.get("languagemodel").path.endswith("llama.onnx")


@pytest.mark.parametrize(
    ("module_path", "module_name"),
    [
        (REPO_ROOT / "python" / "penta_core" / "ml" / "model_registry.py", "root_model_registry_infer"),
        (REPO_ROOT / "music_brain" / "penta_core" / "ml" / "model_registry.py", "music_brain_model_registry_infer"),
    ],
)
def test_task_inference_recognizes_new_model_names(module_path, module_name):
    module = load_module(module_path, module_name)
    registry = module.ModelRegistry()

    assert registry._infer_task(Path("models/audiojepa.onnx")) == module.ModelTask.AUDIO_JEPA
    assert registry._infer_task(Path("models/chord_jepa.onnx")) == module.ModelTask.CHORD_JEPA
    assert registry._infer_task(Path("models/llama_onnx.onnx")) == module.ModelTask.LANGUAGE_MODEL
