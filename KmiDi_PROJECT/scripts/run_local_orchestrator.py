#!/usr/bin/env python3
"""
Local orchestrator runner (Metal/llama.cpp + Tier-1 MIDI)

Responsibilities:
- Load a Mistral 7B GGUF (Q4_K_M/Q5_K_M) via llama.cpp (Metal build).
- Parse free-form intent -> structured intent (BrainController).
- Run deterministic Tier-1 MIDI pipeline.
- Optionally export MIDI to disk (local only).
- Optional branches (images/audio/voice) are off by default.

Usage examples:
  python KmiDi_PROJECT/scripts/run_local_orchestrator.py \
    --gguf-path ~/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf \
    --intent "moody triphop at 90 bpm with fragile vocals" \
    --export-midi ~/Music/iDAW_Output/midi/example.mid

  python KmiDi_PROJECT/scripts/run_local_orchestrator.py \
    --config KmiDi_PROJECT/config/orchestration_local_metal.yaml \
    --intent-file ./prompt.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "source" / "python"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from music_brain.orchestration.local_orchestrator import (  # noqa: E402
    LocalMultiModelOrchestrator,
    OrchestrationConfig,
)


def _load_yaml_config(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - optional dependency
        raise SystemExit("PyYAML is required to load --config") from exc

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _maybe_int(val: Any) -> Optional[int]:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local Mistral (llama.cpp Metal) -> Tier-1 MIDI orchestrator"
    )
    parser.add_argument("--gguf-path", help="Path to Mistral 7B GGUF (Q4_K_M/Q5_K_M).")
    parser.add_argument("--config", help="Optional YAML config for orchestrator.")
    parser.add_argument("--intent", help="Inline intent text.")
    parser.add_argument("--intent-file", help="Path to a text file containing intent.")
    parser.add_argument("--mistral-ctx", type=int, default=3072)
    parser.add_argument("--llama-threads", type=int, help="Override thread count for llama.cpp")
    parser.add_argument("--n-gpu-layers", type=int, help="Override n_gpu_layers for llama.cpp")
    parser.add_argument("--keep-loaded", action="store_true", help="Keep LLM loaded between calls")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--midi-length", type=int, default=16)
    parser.add_argument(
        "--midi-device", default="mps", help='Tier-1 device: "mps", "cpu", or "auto"'
    )
    parser.add_argument(
        "--export-midi",
        help="Path to write MIDI. Defaults to output_root/midi/ if omitted.",
    )
    parser.add_argument(
        "--output-root",
        default=str(Path.home() / "Music" / "iDAW_Output"),
        help="Base output directory for artifacts.",
    )
    parser.add_argument(
        "--enable-images",
        action="store_true",
        help="Enable image branch (disabled by default).",
    )
    parser.add_argument(
        "--enable-audio-texture",
        action="store_true",
        help="Enable audio texture branch (disabled by default).",
    )
    parser.add_argument(
        "--enable-voice",
        action="store_true",
        help="Enable voice branch (disabled by default).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg_data: Dict[str, Any] = {}
    if args.config:
        cfg_data = _load_yaml_config(Path(args.config))

    intent_text = args.intent
    if args.intent_file:
        intent_text = Path(args.intent_file).read_text(encoding="utf-8").strip()
    if not intent_text:
        raise SystemExit("Provide --intent or --intent-file")

    gguf_path = args.gguf_path or cfg_data.get("mistral_gguf_path") or cfg_data.get("gguf_path")
    if not gguf_path:
        raise SystemExit("Missing --gguf-path (or mistral_gguf_path in --config).")

    cfg_mistral_ctx = _maybe_int(cfg_data.get("mistral_ctx", args.mistral_ctx))
    cfg_mistral_ctx = cfg_mistral_ctx if cfg_mistral_ctx is not None else int(args.mistral_ctx)

    cfg_mistral_seed = _maybe_int(cfg_data.get("mistral_seed", args.seed))
    cfg_llama_threads = _maybe_int(cfg_data.get("llama_threads", args.llama_threads))
    cfg_mistral_gpu_layers = _maybe_int(cfg_data.get("mistral_gpu_layers", args.n_gpu_layers))

    cfg_midi_seed = _maybe_int(cfg_data.get("midi_seed", args.seed))
    cfg_midi_seed = cfg_midi_seed if cfg_midi_seed is not None else args.seed

    cfg_midi_length = _maybe_int(cfg_data.get("midi_length", args.midi_length))
    cfg_midi_length = cfg_midi_length if cfg_midi_length is not None else args.midi_length

    cfg = OrchestrationConfig(
        mistral_gguf_path=Path(gguf_path).expanduser(),
        mistral_ctx=cfg_mistral_ctx,
        mistral_seed=cfg_mistral_seed,
        llama_threads=cfg_llama_threads,
        mistral_gpu_layers=cfg_mistral_gpu_layers,
        keep_brain_loaded=bool(cfg_data.get("keep_brain_loaded", args.keep_loaded)),
        enable_images=bool(cfg_data.get("enable_images", args.enable_images)),
        enable_audio_texture=bool(cfg_data.get("enable_audio_texture", args.enable_audio_texture)),
        enable_voice=bool(cfg_data.get("enable_voice", args.enable_voice)),
        output_root=Path(cfg_data.get("output_root", args.output_root)).expanduser(),
        midi_seed=cfg_midi_seed,
        midi_length=cfg_midi_length,
        midi_device=str(cfg_data.get("midi_device", args.midi_device)),
    )

    orchestrator = LocalMultiModelOrchestrator(cfg)
    result = orchestrator.execute(intent_text)

    midi_path = None
    if result.midi:
        export_target = cfg_data.get("export_midi") or args.export_midi
        if export_target:
            target_path = Path(export_target).expanduser()
            midi_path = orchestrator.export_midi(
                result.midi, output_dir=target_path.parent, filename=target_path.name
            )
        else:
            midi_path = orchestrator.export_midi(result.midi)

    summary = {
        "intent_title": result.intent.title,
        "midi_path": str(midi_path) if midi_path else None,
        "images": [str(p) for p in result.image_paths],
        "audio_textures": [str(p) for p in result.audio_textures],
        "voice_outputs": [str(p) for p in result.voice_outputs],
        "errors": result.errors,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
