"""Tests for Phase 5 — Training infrastructure consolidation."""

import json
import sys
import pytest
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

# Load model_registry.py directly to bypass broken penta_core.ml.__init__
# (it uses bare "from penta_core.ml..." imports — pre-existing tech debt)
_spec = importlib.util.spec_from_file_location(
    "model_registry",
    str(REPO_ROOT / "music_brain" / "penta_core" / "ml" / "model_registry.py"),
)
mr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mr)

ModelRegistry = mr.ModelRegistry
ModelInfo = mr.ModelInfo
ModelTask = mr.ModelTask
ModelBackend = mr.ModelBackend
get_registry = mr.get_registry


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset ModelRegistry singleton between tests."""
    ModelRegistry._instance = None
    yield
    ModelRegistry._instance = None


class TestModelRegistry:
    def test_jepa_task_types_exist(self):
        assert ModelTask.AUDIO_JEPA.value == "audio_jepa"
        assert ModelTask.CHORD_JEPA.value == "chord_jepa"

    def test_singleton_pattern(self):
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2

    def test_register_and_get(self):
        registry = ModelRegistry()
        model = ModelInfo(
            name="test_model",
            task=ModelTask.AUDIO_JEPA,
            backend=ModelBackend.PYTORCH,
            path="/tmp/test.pt",
        )
        registry.register(model)
        assert registry.get("test_model") is model
        assert registry.get("nonexistent") is None

    def test_list_by_task(self):
        registry = ModelRegistry()
        registry.register(ModelInfo("a", ModelTask.AUDIO_JEPA, ModelBackend.PYTORCH, "/a.pt"))
        registry.register(ModelInfo("b", ModelTask.CHORD_JEPA, ModelBackend.PYTORCH, "/b.pt"))
        registry.register(ModelInfo("c", ModelTask.EMOTION_CLASSIFICATION, ModelBackend.ONNX, "/c.onnx"))

        jepa = registry.list(ModelTask.AUDIO_JEPA)
        assert len(jepa) == 1
        assert jepa[0].name == "a"

    def test_infer_task_jepa(self):
        registry = ModelRegistry()
        assert registry._infer_task(Path("/checkpoints/audio_jepa/best.pt")) == ModelTask.AUDIO_JEPA
        assert registry._infer_task(Path("/checkpoints/chord_jepa/best.pt")) == ModelTask.CHORD_JEPA
        assert registry._infer_task(Path("/models/emotion_v2.onnx")) == ModelTask.EMOTION_CLASSIFICATION

    def test_project_checkpoints_in_default_dirs(self):
        registry = ModelRegistry()
        dir_strs = [str(d) for d in registry._model_dirs]
        ckpt_dir = REPO_ROOT / "checkpoints"
        if ckpt_dir.exists():
            assert any("checkpoints" in d for d in dir_strs)

    def test_model_info_serialization(self):
        info = ModelInfo(
            name="test",
            task=ModelTask.AUDIO_JEPA,
            backend=ModelBackend.PYTORCH,
            path="/tmp/test.pt",
            tags=["jepa", "test"],
        )
        d = info.to_dict()
        restored = ModelInfo.from_dict(d)
        assert restored.name == info.name
        assert restored.task == info.task
        assert restored.tags == info.tags


class TestRegistryManifest:
    def test_registry_json_valid(self):
        manifest_path = REPO_ROOT / "checkpoints" / "registry.json"
        if not manifest_path.exists():
            pytest.skip("No registry.json")
        with open(manifest_path) as f:
            data = json.load(f)
        assert "models" in data
        assert len(data["models"]) > 0

    def test_load_registry_manifest(self):
        manifest_path = REPO_ROOT / "checkpoints" / "registry.json"
        if not manifest_path.exists():
            pytest.skip("No registry.json")
        registry = ModelRegistry()
        count = registry.load_registry_manifest(str(manifest_path), validate=False)
        assert count > 0
        assert registry.get("chord_jepa_v1") is not None


class TestTrainingScripts:
    def test_train_jepa_exists(self):
        path = REPO_ROOT / "training" / "scripts" / "train_jepa.py"
        assert path.exists()

    def test_train_emotion_exists(self):
        path = REPO_ROOT / "training" / "scripts" / "train_emotion_optimized.py"
        assert path.exists()

    def test_jepa_config_exists(self):
        path = REPO_ROOT / "config" / "jepa_training.yaml"
        assert path.exists()
