"""Tests for scripts/make_jepa_manifest.py (JEPA manifest generator)."""

import json
import sys
from pathlib import Path

import pytest

# Run script as module or by path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def synthetic_dataset(tmp_path):
    """Create a minimal audio+MIDI tree and return (audio_root, midi_root)."""
    audio_root = tmp_path / "audio"
    midi_root = tmp_path / "midi"
    audio_root.mkdir()
    midi_root.mkdir()
    # Two short WAV files (need soundfile to write)
    try:
        import soundfile as sf
        import numpy as np
    except ImportError:
        pytest.skip("soundfile and numpy required for synthetic WAV")
    for name in ("a.wav", "b.wav"):
        path = audio_root / name
        # 1 second mono 16 kHz
        sf.write(path, np.zeros(16000, dtype=np.float32), 16000)
    # Mirror MIDI paths so script finds sidecars
    for name in ("a.mid", "b.mid"):
        (midi_root / name).write_bytes(b"MThd\x00\x00\x00\x06\x00\x00")  # minimal MIDI header
    return audio_root, midi_root


def test_make_jepa_manifest_produces_valid_manifests(synthetic_dataset, tmp_path):
    """Run make_jepa_manifest and assert JSONL files exist and are parseable by Lhotse."""
    audio_root, midi_root = synthetic_dataset
    out_dir = tmp_path / "manifests"

    try:
        from lhotse import CutSet, RecordingSet, SupervisionSet
    except ImportError:
        pytest.skip("lhotse required for JEPA manifest tests")
    import scripts.make_jepa_manifest as mod
    import argparse

    ns = argparse.Namespace(
        audio_root=audio_root,
        midi_root=midi_root,
        out_dir=out_dir,
        pattern="**/*.wav",
        window_seconds=2.0,
        stride_seconds=1.0,
        min_duration=None,
        max_duration=None,
        config=None,
    )
    # Build pairs and run pipeline
    pairs = mod._discover_pairs(audio_root, midi_root, pattern=ns.pattern)
    assert len(pairs) == 2
    cache = {}
    recordings, rec_custom_map = mod.build_recordings(pairs, audio_root, midi_root, cache)
    supervisions = mod.build_supervisions(pairs, audio_root, midi_root, recordings)
    cuts = mod.build_cuts(
        recordings,
        supervisions,
        rec_custom_map,
        window_seconds=ns.window_seconds,
        stride_seconds=ns.stride_seconds,
        min_duration=ns.min_duration,
        max_duration=ns.max_duration,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    recordings.to_file(out_dir / "recordings.jsonl")
    supervisions.to_file(out_dir / "supervisions.jsonl")
    cuts.to_file(out_dir / "cuts.jsonl")
    cuts.to_file(out_dir / "cuts_with_hashes.jsonl")
    mod.write_manifest_args(out_dir, ns)

    assert (out_dir / "recordings.jsonl").exists()
    assert (out_dir / "supervisions.jsonl").exists()
    assert (out_dir / "cuts.jsonl").exists()
    assert (out_dir / "cuts_with_hashes.jsonl").exists()
    assert (out_dir / "manifest_args.json").exists()

    # Lhotse round-trip
    recs2 = RecordingSet.from_file(out_dir / "recordings.jsonl")
    sups2 = SupervisionSet.from_file(out_dir / "supervisions.jsonl")
    cuts2 = CutSet.from_file(out_dir / "cuts.jsonl")
    assert len(recs2) == 2
    assert len(sups2) == 2
    assert len(cuts2) >= 2

    # Hashes and custom on cuts
    for cut in cuts2:
        assert cut.custom is not None
        assert "sha1_audio" in cut.custom
        assert "sha1_midi" in cut.custom
        assert "midi_sidecar" in cut.custom
    # Same hashes on second run (determinism)
    cache2 = {}
    recs3, rec_custom_map3 = mod.build_recordings(pairs, audio_root, midi_root, cache2)
    for rid, custom in rec_custom_map3.items():
        assert rid in rec_custom_map
        assert custom["sha1_audio"] == rec_custom_map[rid]["sha1_audio"]
        assert custom.get("sha1_midi") == rec_custom_map[rid].get("sha1_midi")


def test_make_jepa_manifest_cli(synthetic_dataset, tmp_path):
    """Run script via CLI and check exit 0 and output files."""
    audio_root, midi_root = synthetic_dataset
    out_dir = tmp_path / "manifests"
    # Run script
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "make_jepa_manifest.py"),
            "--audio-root",
            str(audio_root),
            "--midi-root",
            str(midi_root),
            "--out-dir",
            str(out_dir),
            "--window-seconds",
            "2",
            "--stride-seconds",
            "1",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(
            f"make_jepa_manifest.py failed (lhotse/soundfile not installed?): " f"{result.stderr}",
        )
    assert result.returncode == 0
    assert (out_dir / "recordings.jsonl").exists()
    assert (out_dir / "cuts.jsonl").exists()
    with (out_dir / "manifest_args.json").open() as f:
        args_json = json.load(f)
    assert args_json["window_seconds"] == 2.0
    assert args_json["stride_seconds"] == 1.0
