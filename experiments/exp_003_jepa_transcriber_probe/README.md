# exp_003: JEPA-to-MIDI transcriber probe

**Status:** Design / stub. No implementation yet.

## Goal

Implement a lightweight **trainable probe** on top of a **frozen** JEPA encoder (e.g. WavJEPA) that maps latent representations to MIDI/symbolic output (e.g. note events or token sequences). Encoder stays frozen; only the probe head is trained.

## Design

- **Docs:** [docs/WAVJEPA_LATENT_PIPELINE.md](../../docs/WAVJEPA_LATENT_PIPELINE.md), [docs/mt3-transcription-baseline.md](../../docs/mt3-transcription-baseline.md)
- **Source plan:** [docs/SOURCE_INTEGRATION_PLAN.md](../../docs/SOURCE_INTEGRATION_PLAN.md) — briefing “JEPA-to-MIDI transcriber probe implementation”
- **Repo:** `music_brain/jepa/`; design assumes frozen encoder, optional linear map, then token/note head

## Constraints

- Use only **verified, license-cleared** assets (encoder checkpoint, training data).
- No encoder training; determinism and reproducibility per WAVJEPA_LATENT_PIPELINE.
- Probe architecture (linear head, small MLP, or MT3 adapter) and eval protocol to be defined before implementation.

## When to implement

After: (1) encoder and data sources are adopted in `config/source_manifest.yaml`, (2) probe architecture and eval protocol are decided, (3) design doc is updated.
