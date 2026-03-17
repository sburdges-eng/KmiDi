# New MIDI datasets & models

**Item:** New MIDI datasets & models  
**Task:** Integrate "new MIDI datasets & models" into discovery and acquisition plan.  
**Secondary tasks:** Add to config/manifest; align with make_jepa_manifest and prepare_datasets; respect DATA LAW.  
**Context:** MIDI datasets; aligned audio-MIDI; config/dataset_manifest_schema.json; scripts/make_jepa_manifest.py.  
**Verification status:** UNVERIFIED  
**Verification basis:** source_title_only; repo has MAESTRO-like and Lhotse; "new" unspecified; no pasted source.  
**Known facts:**
- Repo uses MAESTRO-like audio+MIDI, Lhotse
- DATA_AND_TRAINING, KMIDI_DATASETS_PATH; DatasetManifest 2.0  
**Unknowns:** Which datasets and models; URLs; licenses; size; alignment format.  
**Assumptions:** None  
**Constraints:** No large data in repo; use env-directed paths.  
**Ambiguities:** "New" vs existing; "models" = tokenizers vs checkpoints vs both.  
**Source text / data:** UNKNOWN (full source block not present)  
**Output format:** Manifest entries; docs; config and scripts; storage outside git.
