# JEPA-to-MIDI transcriber probe implementation

**Item:** JEPA-to-MIDI transcriber probe implementation  
**Task:** Plan implementation of a JEPA-to-MIDI probe (latent → MIDI/symbolic) with explicit decoding strategy options.  
**Secondary tasks:** Tie to WavJEPA latent pipeline and MT3 baseline; document placement, determinism, and whether draft-conditioned constrained decoding is a better fit than naive token masking.  
**Context:** JEPA/WavJEPA/transcriber; docs/WAVJEPA_LATENT_PIPELINE.md, mt3-transcription-baseline.md; music_brain/jepa.  
**Verification status:** PARTIALLY_VERIFIED  
**Verification basis:** repo_scan_only + user_pasted_briefing; WAVJEPA_LATENT_PIPELINE and MT3 exist in repo, but no probe implementation exists yet.  
**Known facts:**
- WAVJEPA_LATENT_PIPELINE: Audio → frozen WavJEPA → latents → optional linear map → token head
- MT3 as token-decoder baseline
- No probe implementation in repo
- User briefing describes draft-conditioned constrained decoding (DCCD) as a training-free, two-stage inference pattern: unconstrained semantic draft first, grammar/trie-constrained final decode second.
- User briefing positions REMI+BPE tokenization, best-of-K draft sampling, and grammar/trie masks as a practical path for valid symbolic outputs on small or quantized models.
- The repo already has REMI+BPE context (`docs/REMI_BPE_TOKENIZATION.md`, `experiments/perch_remi_pipeline`) that can anchor a constrained symbolic decode experiment.
**Unknowns:** Probe architecture; training data; eval protocol; whether the final stage should emit tokenized REMI, direct MIDI events, or a stricter intermediate grammar; probe = trainable head only or draft+decoder pair?  
**Assumptions:** ASSUMPTION: the probe should keep the JEPA encoder frozen and concentrate complexity in the symbolic decoder path.  
**Constraints:** Determinism and no encoder training per WAVJEPA_LATENT_PIPELINE; constrained decode must preserve valid grammar/token structure without introducing hidden runtime services.  
**Ambiguities:** Probe vs full transcriber; which JEPA encoder; how draft text/plan is represented for music; whether constrained decode should be trie-based, grammar-based, or both.  
**Source text / data:** User briefing pasted in full; no primary-source paper or repo URL attached yet.  
**Captured guidance:**
- Standard masked-from-token-0 decoding may distort semantic intent; a draft-then-constrain path may be better for valid MIDI/token generation.
- Best-of-K draft selection is a plausible stability lever for small on-device models.
- Grammar/trie-constrained decoding should be evaluated as a symbolic validity mechanism, not just as a syntax afterthought.
**Output format:** Design doc; experiment under `experiments/`; config for data/eval; optional grammar/trie spec for constrained symbolic decode.
