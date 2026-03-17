# StructXLIP: Plugging into LALM Fine-Tuning

StructXLIP provides **Audio Edge Maps** (onset envelope, spectral flux) and **structure-centric alignment losses** so the LALM can ground temporal instructions (e.g. "drop at bar 16, beat 3") instead of relying only on dense mel–text alignment.

## Components

- **`penta_core.ml.structxlip.audio_edges`**: `extract_audio_edge_maps(y, sr, ...)` returns `{"onset_envelope", "spectral_flux"}` (1D arrays per time frame). Use these as structural proxies in your dataloader.
- **`penta_core.ml.structxlip.structure_losses`**: `global_structure_loss(audio_edge_features, text_structure_features)` — contrastive (InfoNCE) alignment. `local_structure_loss` and `consistency_edge_loss` are placeholders for future use.

## Dataloader

In your LALM fine-tuning pipeline:

1. Load waveform (or use existing mel path).
2. Call `extract_audio_edge_maps(y, sr=sr, hop_length=512)` to get onset and flux.
3. Pool or project edge maps to a fixed-size vector (e.g. mean over time, or a small CNN) to get `audio_edge_features` of shape `(batch, embed_dim)`.
4. Encode structural text (e.g. "drop at beat 3", "chorus starts at bar 16") with your text encoder to get `text_structure_features` of shape `(batch, embed_dim)`.
5. Add `global_structure_loss(edge_features, text_features)` to your total loss with a weight (e.g. 0.1–0.3). Tune so the main alignment loss still dominates.

## Loss weights

- Start with a small weight for `global_structure_loss` (e.g. 0.1) so the model does not collapse to structure-only.
- When `local_structure_loss` and `consistency_edge_loss` are implemented, add them with separate weights; the consistency term ties edge proxies back to the continuous audio representation to avoid drift.

## References

- StructXLIP-style alignment: structural proxies (e.g. edge maps) aligned with filtered structural text to stabilize vision–language alignment; here adapted to audio for LALM.
