# JEPA Manifest Generator — DataLad Workflow

Reproducible JEPA manifests for audio+MIDI datasets (e.g. MAESTRO) using [Lhotse](https://github.com/lhotse-speech/lhotse) and [DataLad](https://www.datalad.org/). The generator script writes `recordings.jsonl`, `supervisions.jsonl`, `cuts.jsonl`, and `cuts_with_hashes.jsonl` with optional SHA1 provenance.

## Dependencies

```bash
pip install -e ".[jepa]"   # lhotse + soundfile
# Optional, for provenance:
pip install datalad
```

## 1. Dataset repo creation

Create a DataLad dataset (e.g. for a MAESTRO subset or derived manifest repo):

```bash
datalad create kmidi-mini-maestro
cd kmidi-mini-maestro
mkdir manifests
```

## 2. Generate manifests (recorded run)

From the **KmiDi repo root**, run the script and record the exact command with DataLad so provenance is tracked:

```bash
datalad run -m "Generate JEPA manifests (8s windows)" \
  "python scripts/make_jepa_manifest.py \
   --audio-root /path/to/maestro/audio \
   --midi-root /path/to/maestro/midi \
   --out-dir manifests \
   --window-seconds 8.0 \
   --stride-seconds 4.0"
```

Use paths appropriate for your machine (e.g. `~/Datasets/maestro-v3/maestro-v3.0.0` for audio and MIDI). DataLad will store the command in the commit message and link the output files.

## 3. Save and version

```bash
datalad save -m "Add Lhotse manifests for JEPA windows"
```

For different window configs, re-run with new parameters and commit again:

```bash
datalad run -m "Regenerate 4s windows" \
  "python scripts/make_jepa_manifest.py \
   --audio-root /path/to/maestro/audio \
   --midi-root /path/to/maestro/midi \
   --out-dir manifests/v2_4s \
   --window-seconds 4.0 \
   --stride-seconds 2.0"
```

## 4. Versioning convention

- Use a subdir per config if you keep multiple variants, e.g. `manifests/v1_8s/`, `manifests/v2_4s/`.
- Tag after a stable generation: `git tag manifests-8s-v1`.

## 5. Output files

| File | Description |
|------|-------------|
| `recordings.jsonl` | Lhotse RecordingSet (one row per audio file) |
| `supervisions.jsonl` | Lhotse SupervisionSet (piece/segment metadata, midi_sidecar) |
| `cuts.jsonl` | Lhotse CutSet (fixed-window segments for JEPA) |
| `cuts_with_hashes.jsonl` | Same as cuts with `sha1_audio` and `sha1_midi` in each cut's custom dict |
| `manifest_args.json` | CLI arguments used for reproducibility |

## References

- [DATA_AND_TRAINING.md](DATA_AND_TRAINING.md) — dataset paths and training governance
- [docs/research/MULTIMODAL_IMPLEMENTATIONS_PLAN.md](research/MULTIMODAL_IMPLEMENTATIONS_PLAN.md) — step 2 (JEPA manifest generator)
- [MULTIMODAL_REPRESENTATIONS_2026.md](research/MULTIMODAL_REPRESENTATIONS_2026.md) — manifest schema and JEPA context
