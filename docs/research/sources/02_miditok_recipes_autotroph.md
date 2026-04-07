# MidiTok recipes for Autotroph tokenizers

**Item:** MidiTok recipes for Autotroph tokenizers  
**Task:** Capture MidiTok recipes for "Autotroph tokenizers" and separate public symbolic-tokenization facts from the undefined "Autotroph" label.  
**Secondary tasks:** Align with REMI-BPE (`perch_remi_pipeline`, `REMI_BPE_TOKENIZATION.md`); map dataset-level config trends that may eventually feed Autotroph if it becomes a real public config surface.  
**Context:** Symbolic tokenization; MidiTok; Autotroph (KmiDi term; no repo refs).  
**Verification status:** PARTIALLY_VERIFIED  
**Verification basis:** repo_scan_only + user_pasted_briefing; MidiTok is in repo, but Autotroph still has no repo or public-source grounding in this workspace.  
**Known facts:**
- Repo uses MidiTok REMI+BPE
- experiments/perch_remi_pipeline; docs/REMI_BPE_TOKENIZATION.md
- User briefing says recent symbolic-music configuration is increasingly dataset-driven, with externalized filter configs, heuristic feature filters, and prompt/alignment schemas rather than only model hyperparameters.
- User briefing names Aria-MIDI, GigaMIDI, MIDI-RWKV, and MIDI-LLaMA as relevant public references around filtering configs, PEFT/state tuning, and multimodal prompt alignment.
- Autotroph is still not referenced anywhere in repo-local scans.
- User briefing explicitly reports no verifiable public dataset, model, or tooling signal for Autotroph in the MIDI/symbolic space.
**Unknowns:** What "Autotroph tokenizers" are; which recipes apply; whether Autotroph is an internal codename, a planned config family, a vocabulary, or a separate unpublished codebase.  
**Assumptions:** None  
**Constraints:** Keep tokenization configs deterministic, versioned, and externalized; do not add speculative Autotroph code paths without a concrete schema or source.  
**Ambiguities:** Recipes vs Autotroph; public vs internal; tokenizer config vs higher-level dataset curation config.  
**Source text / data:** User briefing pasted in full; no primary-source verification attached for Autotroph itself.  
**Captured guidance:**
- Treat REMI+BPE via MidiTok as the concrete, public tokenizer path today.
- Treat Autotroph as unresolved vocabulary until a real contract, schema, or upstream source is attached.
- Expect future tokenization work to include dataset-level configs, PEFT/state-tuning metadata, and prompt/alignment configuration, not just vocab settings.
**Output format:** Doc (recipes, vocab, config boundaries); optional experiment/config only after Autotroph is defined; manifest entries only when public sources and licenses are attached.
