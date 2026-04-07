# New MIDI datasets & models

**Item:** New MIDI datasets & models  
**Task:** Integrate recent symbolic-music datasets, model/tool candidates, and preprocessing trends into discovery and acquisition planning.  
**Secondary tasks:** Add to config/manifest; align with `make_jepa_manifest` and dataset-prep flows; capture tool-vs-dataset boundaries; respect DATA LAW.  
**Context:** MIDI datasets; aligned audio-MIDI; config/dataset_manifest_schema.json; scripts/make_jepa_manifest.py.  
**Verification status:** PARTIALLY_VERIFIED  
**Verification basis:** repo_scan_only + user_pasted_briefing; repo has symbolic tokenization and dataset docs, and the new item now includes a full pasted briefing, but primary-source URLs/licenses still need per-item verification.  
**Known facts:**
- Repo uses MAESTRO-like audio+MIDI, Lhotse
- DATA_AND_TRAINING, KMIDI_DATASETS_PATH; DatasetManifest 2.0
- User briefing highlights Aria-MIDI as a large MIDI pipeline with explicit light/heavy filtering configs and deduplication modes.
- User briefing highlights GigaMIDI-derived expressiveness heuristics such as velocity/timing/alignment measures as a data-filtering layer, not just a post-hoc analysis trick.
- User briefing highlights MIDI-RWKV as evidence that state tuning and LoRA-style PEFT are now part of the symbolic-model tuning story.
- User briefing highlights MIDI-LLaMA-style multimodal alignment as a config layer involving prompts, alignment objectives, and cross-modal token mapping.
- User briefing highlights Beautiful-Motifs as a lightweight motif-extraction tool for pulling short symbolic seeds from MIDI into standalone motif files.
**Unknowns:** Exact URLs, licenses, sizes, and redistribution terms for each candidate; which items are datasets vs tools vs model checkpoints; whether any candidate is production-safe for commercial or closed redistribution.  
**Assumptions:** Treat these as briefing-derived candidates until each item has a manifest entry with license, source, and storage policy.  
**Constraints:** No large data in repo; use env-directed paths; no mutable manifests; do not mix freeze-lane and exploratory data without explicit tagging.  
**Ambiguities:** "Models" = tokenizers vs checkpoints vs alignment frameworks vs extractors; whether Beautiful-Motifs belongs in corpus prep, conditioning assets, or post-processing.  
**Source text / data:** User briefing pasted in full; primary-source verification still required item by item.  
**Captured guidance:**
- Dataset-level filtering configs are now a first-class reproducibility surface.
- Feature-derived heuristics can be used as upstream curation signals, not just downstream metrics.
- Motif extraction belongs on the symbolic-data-tooling watchlist because it can generate reusable micro-seed libraries for conditioning and retrieval.
- PEFT/state-tuning and multimodal prompt-alignment configs should be tracked alongside tokenizer and dataset configs.
**Output format:** Manifest entries; docs; config and scripts; storage outside git; explicit source/license/hash capture before adoption.
