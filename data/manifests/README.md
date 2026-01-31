# Data Manifests

JEPA training configs (`configs/jepa_*.yaml`) reference `data/manifests/aligned.jsonl`. penta_core discovers models from `registry.json` (see [docs/AI_MODEL_STRUCTURES.md](../docs/AI_MODEL_STRUCTURES.md)).

## registry.json / registry.schema.json

- **registry.json** — Model registry manifest for penta_core `model_registry.load_registry_manifest()`. Paths point to `~/Models` (weights never in repo).
- **registry.schema.json** — JSON schema for `registry.json`; optional validation when `jsonschema` is installed.

## aligned.jsonl format

One JSON object per line. Each row describes an aligned triple (audio, MIDI, Spectocloud):

```json
{"audio_path": "/path/to/audio.wav", "midi_path": "/path/to/midi.mid", "specto_path": "/path/to/specto.npy", "start_offset": 0.0, "tempo": 120, "timebase": 480}
```

| Field | Type | Description |
|-------|------|-------------|
| audio_path | str | Path to audio file (.wav, .aiff, .mp3) |
| midi_path | str | Path to MIDI file |
| specto_path | str | Path to precomputed Spectocloud tensor (.npy) or rendered image |
| start_offset | float | Start offset in seconds (for alignment) |
| tempo | float | BPM |
| timebase | int | PPQ / timebase |

Symlink `data/` to `~/Datasets/kmidi_jepa/` or set manifest path in experiment config to point at your dataset.
