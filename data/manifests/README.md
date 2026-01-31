# Data Manifests

JEPA training configs (`configs/jepa_*.yaml`) reference `data/manifests/aligned.jsonl`.

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
