# WavJEPA pretrained waveform encoder released

**Item:** WavJEPA pretrained waveform encoder released  
**Task:** Integrate and plan acquisition of "WavJEPA pretrained waveform encoder" release.  
**Secondary tasks:** Use as frozen front-end per WAVJEPA_LATENT_PIPELINE; document checkpoint and license.  
**Context:** WavJEPA; exp_002 references labhamlet/wavjepa-base (HF).  
**Verification status:** PARTIALLY_VERIFIED  
**Verification basis:** repo_scan_only + source_title_only; WAVJEPA_KMIDI_TASKS and exp_002 reference HF; no pasted source.  
**Known facts:**
- Repo references loading WavJEPA (e.g. Hugging Face)
- Design uses frozen encoder only  
**Unknowns:** Official URL; license; exact artifact (HF id, files, checksums); public vs gated.  
**Assumptions:** ASSUMPTION: "released" = at least one publicly loadable checkpoint (UNVERIFIED).  
**Constraints:** Use only as frozen feature extractor.  
**Ambiguities:** Variant (base/large); 16 kHz / 2 s compatibility.  
**Source text / data:** UNKNOWN (full source block not present)  
**Output format:** Manifest entry; doc; download to KELLY_MODELS_PATH or env path; config for model id.
