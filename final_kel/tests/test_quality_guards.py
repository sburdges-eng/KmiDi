import sys
import types
import importlib
from argparse import Namespace

import numpy as np
import pytest


def test_kellybrain_accepts_non_tensor_features():
    torch = pytest.importorskip("torch")
    from Brain.kellybrain.core import KellyBrain

    class DummyEmotion(torch.nn.Module):
        def forward(self, z):
            if z.dim() == 1:
                z = z.unsqueeze(0)
            return z

    class DummyChords(torch.nn.Module):
        def forward(self, z):
            if z.dim() == 2:
                z = z.unsqueeze(1)
            return torch.zeros((z.shape[0], 1, 12))

    brain = KellyBrain(
        audio_model=None,
        chord_model=DummyChords(),
        emotion_model=DummyEmotion(),
    )
    out = brain.think([0.1] * 256)
    assert "next_chord" in out
    assert "mood_vector" in out


def test_dataset_pipeline_writes_single_external_dataset_file(tmp_path):
    from training.dataset_pipeline import run_pipeline

    tabs = tmp_path / "song.txt"
    tabs.write_text("E|----\n", encoding="utf-8")
    out_file = tmp_path / "external" / "kmidi_multimodal.npy"
    data = run_pipeline(str(tmp_path), str(out_file), save=True)

    entry = data[str(tabs)]
    assert entry["type"] == "tabs"
    assert "tabs" in entry
    assert out_file.is_file()

    loaded = np.load(out_file, allow_pickle=True).item()
    assert str(tabs) in loaded


def test_storage_paths_honors_external_root_env(monkeypatch, tmp_path):
    import training.storage_paths as storage_paths

    monkeypatch.setenv("KMIDI_EXTERNAL_ROOT", str(tmp_path / "ext_drive"))
    storage_paths = importlib.reload(storage_paths)
    assert storage_paths.EXTERNAL_ROOT == str((tmp_path / "ext_drive").resolve())
    assert storage_paths.PROCESSED_DATASET_FILE.startswith(storage_paths.EXTERNAL_ROOT)


def test_cmd_release_passes_out_dir(monkeypatch, tmp_path):
    from scripts import kmidi_cli

    called = {}

    class FakePack:
        def build(self, out_dir="release"):
            called["out_dir"] = out_dir
            return out_dir

    fake_module = types.SimpleNamespace(DeploymentPack=FakePack)
    monkeypatch.setitem(sys.modules, "tools.artist_stubs", fake_module)

    args = Namespace(out=str(tmp_path / "release"))
    kmidi_cli.cmd_release(args)

    assert called["out_dir"] == str(tmp_path / "release")


def test_cmd_generate_torch_load_fallback(monkeypatch, tmp_path):
    from scripts import kmidi_cli

    ckpt_path = tmp_path / "dummy.pt"
    ckpt_path.write_text("x", encoding="utf-8")
    calls = []

    def fake_load(path, map_location=None, **kwargs):
        calls.append(kwargs)
        if "weights_only" in kwargs:
            raise TypeError("weights_only unsupported")
        return {"state_dict": {}}

    fake_torch = types.SimpleNamespace(load=fake_load)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    args = Namespace(checkpoint=str(ckpt_path), prompt="", out="generated")
    kmidi_cli.cmd_generate(args)

    assert any("weights_only" in kwargs for kwargs in calls)
    assert any("weights_only" not in kwargs for kwargs in calls)
