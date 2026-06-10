#!/usr/bin/env python3
"""
Generate reproducible JEPA manifests for MAESTRO-like audio+MIDI datasets using Lhotse.

Produces RecordingSet, SupervisionSet, and CutSet JSONL files with sha1_audio/sha1_midi
and custom fields (midi_sidecar, emotion_label, bpm). Designed for use with DataLad
for provenance tracking.

Usage (from repo root):
  pip install -e ".[jepa]"   # or: pip install lhotse soundfile
  python scripts/make_jepa_manifest.py --audio-root ~/Datasets/maestro/audio \\
    --midi-root ~/Datasets/maestro/midi --out-dir manifests --window-seconds 8 --stride-seconds 4

See docs/DATA_AND_TRAINING.md and docs/research/MULTIMODAL_IMPLEMENTATIONS_PLAN.md.

Optional: --source-manifest and --list-from-manifest use config/source_manifest.yaml to list
adopted dataset-like sources (adoption_decision=adopted) that could drive --audio-root/--midi-root
once paths are set. See docs/SOURCE_INTEGRATION_PLAN.md Phase 4.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Optional

# Lhotse is optional; fail with a clear message if not installed.
try:
    from lhotse import CutSet, Recording, RecordingSet, SupervisionSegment, SupervisionSet
except ImportError as e:
    print(
        "lhotse is required. Install with: pip install -e '.[jepa]' or pip install lhotse",
        file=sys.stderr,
    )
    raise SystemExit(1) from e

# Audio duration/sampling read via soundfile when needed
try:
    import soundfile as sf
except ImportError:
    sf = None  # type: ignore[assignment]

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


def _sha1_path(path: Path, chunk_kb: int = 8192) -> str:
    """Streaming SHA1 of a file."""
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            data = f.read(chunk_kb * 1024)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def _duration_seconds(path: Path) -> float:
    """Duration in seconds of an audio file."""
    if sf is None:
        raise ImportError("soundfile required for duration; pip install soundfile")
    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)


def _discover_pairs(
    audio_root: Path,
    midi_root: Path,
    pattern: str = "**/*.wav",
    midi_suffix: str = ".mid",
) -> list[tuple[Path, Optional[Path]]]:
    """Discover (audio_path, midi_path) pairs. midi_path may be None if not found."""
    audio_root = audio_root.resolve()
    midi_root = midi_root.resolve()
    pairs: list[tuple[Path, Optional[Path]]] = []
    for apath in sorted(audio_root.glob(pattern)):
        if not apath.is_file():
            continue
        rel = apath.relative_to(audio_root)
        # Mirror path under midi_root: same relative path with extension swapped
        midi_rel = rel.with_suffix(midi_suffix)
        mpath = midi_root / midi_rel
        pairs.append((apath, mpath if mpath.exists() else None))
    return pairs


def _recording_id(audio_path: Path, audio_root: Path) -> str:
    """Stable recording ID from relative path (no spaces, extension stripped)."""
    rel = audio_path.resolve().relative_to(audio_root.resolve())
    return str(rel.with_suffix("")).replace(" ", "_").replace("/", "_")


def build_recordings(
    pairs: list[tuple[Path, Optional[Path]]],
    audio_root: Path,
    midi_root: Path,
    cache_sha1: dict[str, str],
) -> tuple[RecordingSet, dict[str, dict[str, Any]]]:
    """Build RecordingSet. Returns (RecordingSet, recording_id -> custom dict with sha1_*)."""
    recordings = []
    rec_custom_map: dict[str, dict[str, Any]] = {}
    for apath, mpath in pairs:
        rid = _recording_id(apath, audio_root)
        if str(apath) not in cache_sha1:
            cache_sha1[str(apath)] = _sha1_path(apath)
        sha1_audio = cache_sha1[str(apath)]
        sha1_midi: Optional[str] = None
        if mpath is not None:
            if str(mpath) not in cache_sha1:
                cache_sha1[str(mpath)] = _sha1_path(mpath)
            sha1_midi = cache_sha1[str(mpath)]

        custom: dict[str, Any] = {"sha1_audio": sha1_audio}
        if sha1_midi is not None:
            custom["sha1_midi"] = sha1_midi
        rec_custom_map[rid] = custom

        try:
            rec = Recording.from_file(apath, recording_id=rid)
        except Exception:
            if sf is None:
                raise
            dur = _duration_seconds(apath)
            sr = sf.info(str(apath)).samplerate
            from lhotse.audio.source import AudioSource

            rec = Recording(
                id=rid,
                sources=[AudioSource(type="file", channels=[0], source=str(apath))],
                sampling_rate=sr,
                num_samples=int(dur * sr),
                duration=dur,
                channel_ids=[0],
            )
        recordings.append(rec)
    return RecordingSet.from_recordings(recordings), rec_custom_map


def build_supervisions(
    pairs: list[tuple[Path, Optional[Path]]],
    audio_root: Path,
    midi_root: Path,
    recordings: RecordingSet,
) -> SupervisionSet:
    """Build SupervisionSet: one segment per recording with midi_sidecar and optional bpm."""
    segments = []
    for apath, mpath in pairs:
        rid = _recording_id(apath, audio_root)
        rec = next((r for r in recordings if r.id == rid), None)
        if rec is None:
            continue
        duration = rec.duration
        midi_sidecar = str(mpath.relative_to(midi_root)) if mpath else None
        custom: dict[str, Any] = {}
        if midi_sidecar is not None:
            custom["midi_sidecar"] = midi_sidecar
        seg = SupervisionSegment(
            id=f"{rid}-sup0",
            recording_id=rid,
            start=0.0,
            duration=duration,
            channel=0,
            custom=custom if custom else None,
        )
        segments.append(seg)
    return SupervisionSet.from_segments(segments)


def build_cuts(
    recordings: RecordingSet,
    supervisions: SupervisionSet,
    rec_custom_map: dict[str, dict[str, Any]],
    window_seconds: float,
    stride_seconds: float,
    min_duration: Optional[float],
    max_duration: Optional[float],
) -> CutSet:
    """Build CutSet with fixed windows; propagate custom (sha1, midi_sidecar) onto each cut."""
    from lhotse import MonoCut

    cuts_list = []
    for rec in recordings:
        rid = rec.id
        dur = rec.duration
        if min_duration is not None and dur < min_duration:
            continue
        if max_duration is not None and dur > max_duration:
            continue
        rec_custom = rec_custom_map.get(rid, {})
        sups = [s for s in supervisions if s.recording_id == rid]
        start = 0.0
        idx = 0
        while start + window_seconds <= dur or (start < dur and idx == 0):
            end = min(start + window_seconds, dur)
            seg_dur = end - start
            cut_id = f"{rid}_w{idx:05d}"
            custom = dict(rec_custom)
            if sups:
                s = sups[0]
                if getattr(s, "custom", None):
                    custom.setdefault("midi_sidecar", s.custom.get("midi_sidecar"))
                    custom.setdefault("emotion_label", s.custom.get("emotion_label"))
                    custom.setdefault("bpm", s.custom.get("bpm"))
            mono = MonoCut(
                id=cut_id,
                start=start,
                duration=seg_dur,
                channel=0,
                recording=rec,
                supervisions=[],
                custom=custom if custom else None,
            )
            cuts_list.append(mono)
            start += stride_seconds
            idx += 1
            if start >= dur:
                break
    return CutSet.from_cuts(cuts_list)


def list_adopted_jepa_sources(manifest_path: Path) -> int:
    """List source_manifest entries with adoption_decision=adopted that are
    JEPA/dataset-relevant."""
    if yaml is None:
        print("yaml required for --list-from-manifest; pip install pyyaml", file=sys.stderr)
        return 1
    if not manifest_path.is_file():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        return 1
    with manifest_path.open() as f:
        data = yaml.safe_load(f) or {}
    sources = data.get("sources") or []
    jepa_domain = ("jepa", "midi", "wavjepa", "transcriber")
    adopted = [
        s
        for s in sources
        if (s.get("adoption_decision") == "adopted")
        and (
            any(d in (s.get("integration_domain") or "").lower() for d in jepa_domain)
            or "dataset" in (s.get("artifact_classes") or [])
        )
    ]
    if not adopted:
        print("No adopted JEPA/dataset sources in manifest.", file=sys.stderr)
        print(
            "Set adoption_decision: adopted and add proposed_storage_path for dataset-like items.",
            file=sys.stderr,
        )
        return 0
    for s in adopted:
        path = s.get("proposed_storage_path") or "(none)"
        env = s.get("storage_env_var") or "(none)"
        print(f"  {s.get('source_item', '?')}: proposed_storage_path={path} storage_env_var={env}")
    return 0


def write_manifest_args(out_dir: Path, args: argparse.Namespace) -> None:
    """Write manifest_args.json for reproducibility."""
    d = {k: getattr(args, k) for k in dir(args) if not k.startswith("_") and k != "config"}
    for key in ("audio_root", "midi_root", "out_dir", "config"):
        if key in d and d[key] is not None:
            d[key] = str(Path(d[key]).resolve()) if d[key] else None
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "manifest_args.json").open("w") as f:
        json.dump(d, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate JEPA manifests (Lhotse RecordingSet, SupervisionSet, CutSet) "
            "for audio+MIDI datasets.",
        ),
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=None,
        help="Root directory of audio files (e.g. MAESTRO WAV); required unless "
        "--list-from-manifest",
    )
    parser.add_argument(
        "--midi-root",
        type=Path,
        default=None,
        help="Root directory of MIDI sidecars; required unless --list-from-manifest",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("manifests"),
        help="Output directory for JSONL manifests",
    )
    parser.add_argument(
        "--pattern",
        default="**/*.wav",
        help="Glob pattern for audio under audio-root",
    )
    parser.add_argument(
        "--midi-suffix",
        default=".mid",
        help="MIDI file extension (default: .mid; use .midi for MAESTRO v3)",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=8.0,
        help="JEPA window length in seconds",
    )
    parser.add_argument(
        "--stride-seconds",
        type=float,
        default=4.0,
        help="Stride between windows in seconds",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=None,
        help="Skip recordings shorter than this (seconds)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Skip recordings longer than this (seconds)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config (CLI overrides)",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=None,
        help="Path to source_manifest.yaml (used with --list-from-manifest)",
    )
    parser.add_argument(
        "--list-from-manifest",
        action="store_true",
        help="List adopted JEPA/dataset sources from source_manifest and exit",
    )
    args = parser.parse_args()

    if args.list_from_manifest:
        manifest = (
            args.source_manifest
            or Path(__file__).resolve().parent.parent / "config" / "source_manifest.yaml"
        )
        return list_adopted_jepa_sources(manifest)

    if args.audio_root is None or args.midi_root is None:
        print(
            "Error: --audio-root and --midi-root are required (or use --list-from-manifest).",
            file=sys.stderr,
        )
        return 1
    audio_root = args.audio_root.resolve()
    midi_root = args.midi_root.resolve()
    out_dir = args.out_dir.resolve()
    if not audio_root.is_dir():
        print(f"Error: --audio-root is not a directory: {audio_root}", file=sys.stderr)
        return 1
    if not midi_root.is_dir():
        print(f"Error: --midi-root is not a directory: {midi_root}", file=sys.stderr)
        return 1

    pairs = _discover_pairs(
        audio_root,
        midi_root,
        pattern=args.pattern,
        midi_suffix=args.midi_suffix,
    )
    if not pairs:
        print("No audio files found.", file=sys.stderr)
        return 1

    cache_sha1: dict[str, str] = {}
    recordings, rec_custom_map = build_recordings(pairs, audio_root, midi_root, cache_sha1)
    supervisions = build_supervisions(pairs, audio_root, midi_root, recordings)
    cuts = build_cuts(
        recordings,
        supervisions,
        rec_custom_map,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    recordings.to_file(out_dir / "recordings.jsonl")
    supervisions.to_file(out_dir / "supervisions.jsonl")
    cuts.to_file(out_dir / "cuts.jsonl")
    # cuts_with_hashes: same as cuts but we ensure custom has sha1_*; already there
    cuts.to_file(out_dir / "cuts_with_hashes.jsonl")
    write_manifest_args(out_dir, args)

    print(
        f"Wrote {len(recordings)} recordings, {len(supervisions)} supervisions, "
        f"{len(cuts)} cuts to {out_dir}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
