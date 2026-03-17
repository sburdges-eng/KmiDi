# JEPA-to-MIDI transcriber probe implementation

**Item:** JEPA-to-MIDI transcriber probe implementation  
**Task:** Plan implementation of "JEPA-to-MIDI transcriber probe" (latent → MIDI/symbolic).  
**Secondary tasks:** Tie to WavJEPA latent pipeline and MT3 baseline; document placement and determinism.  
**Context:** JEPA/WavJEPA/transcriber; docs/WAVJEPA_LATENT_PIPELINE.md, mt3-transcription-baseline.md; music_brain/jepa.  
**Verification status:** PARTIALLY_VERIFIED  
**Verification basis:** repo_scan_only + source_title_only; WAVJEPA_LATENT_PIPELINE and MT3 in repo; no probe impl; no pasted source.  
**Known facts:**
- WAVJEPA_LATENT_PIPELINE: Audio → frozen WavJEPA → latents → optional linear map → token head
- MT3 as token-decoder baseline
- No probe implementation in repo  
**Unknowns:** Probe architecture; training data; eval protocol; probe = trainable head only?  
**Assumptions:** ASSUMPTION: probe = lightweight trainable head on frozen JEPA (not re-training encoder).  
**Constraints:** Determinism and no encoder training per WAVJEPA_LATENT_PIPELINE.  
**Ambiguities:** Probe vs full transcriber; which JEPA encoder.  
**Source text / data:** UNKNOWN (full source block not present)  
**Output format:** Design doc; experiment under experiments/; config for data and eval.
