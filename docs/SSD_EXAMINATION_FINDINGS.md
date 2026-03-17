# SSD volume examination findings (KmiDi-focused)

**Scope:** Read-only examination of `/Volumes/Sean's SSD` paths. **Code only** — audio/media files were ignored. **Tip of dirs** — shallow listing and code/config/doc sampling (no deep traversal). Primary project: KmiDi only.

**Plan reference:** [SSD volume examination for KmiDi](.cursor/plans/ssd_volume_examination_for_kmidi_49910ea6.plan.md) (path map and execution approach).

---

## 1. Improvements

### 1.1 Code

- **Emotion_Instrument_Library path:** Workspace uses a single **Google Drive** path in `KmiDi_FINAL/docs/Emotion_Instrument_Library.md` and `KmiDi_FINAL/assets/emotion_instrument_library_catalog.json` (and `KmiDi_FINAL/shared/data/emotion_instrument_library_catalog.json`). The SSD has `/Volumes/Sean's SSD/Emotion_Instrument_Library` (base/sub). Add an env override (e.g. `EMOTION_INSTRUMENT_LIBRARY_ROOT`) so the catalog or loader can use the SSD path when present; document the SSD path as an alternative in the Emotion_Instrument_Library doc.
- **prepare_datasets.py:** Script has no `m4singer` or `maestro_v3` dataset config; SSD `audio/` has both. Consider adding configs for these if they are first-class datasets, or document that they are manually placed and not managed by prepare_datasets.

### 1.2 Documentation

- **VOLUME_README.md (on SSD):** It describes top-level **KmiDi/** as "KmiDi project copy (standalone on volume)." In fact that folder is a **home-directory backup** (`.claude`, `.cargo`, `.idaw`, etc.), not the repo. Update VOLUME_README to state that **KmiDi** at volume root = home backup; the repo clone is **Dev/KmiDi**. Optionally add the same clarification to a doc in the workspace (e.g. `docs/SSD_WORKDIR_STRUCTURE.md` or `docs/ENVIRONMENT.md`).
- **Dataset path table:** `docs/DATASETS_PREPARE_SCRIPT.md` and `docs/EXTERNAL_DRIVE_AUDIO_FILES.md` already use `KMIDI_DATA_ROOT` and "Sean's SSD" as example. Add a single table or section that lists: volume name(s) (Sean's SSD vs KmiDi-external), canonical dataset root (`$KMIDI_DATA_ROOT/Datasets` or `~/Datasets`), and that `audio/` at volume root is a sibling to `Datasets/` and may hold manually placed datasets (emotion_*, m4singer, maestro, etc.) not necessarily under prepare_datasets output.

### 1.3 Automation

- **consolidate_external_on_ssd.sh:** Uses `Sean's SSD` only; no change needed for that volume. Docs that reference `*EXTERNAL` paths (e.g. after consolidation) should state that post-consolidation the volume has no EXTERNAL suffix; see Naming below.
- **prepare_datasets defaults:** `_detect_default_root()` does not add a candidate for `Sean's SSD` explicitly; it relies on `KMIDI_DATA_ROOT` env. Doc already says "Sean's SSD/Datasets" as example; keep env as single knob.

### 1.4 Data / governance

- **Datasets sections:** SSD `Datasets/` has both numbered sections (00_catalog, 20_audio_music, …) and canonical sections (by_domain, by_source, staging, `_index`). README on SSD says legacy numbered dirs can be removed; content moved into canonical sections. Align workspace docs (e.g. `docs/EXTERNAL_DRIVE_AUDIO_FILES.md`, `docs/SSD_WORKDIR_STRUCTURE.md`) so they mention the canonical layout (by_domain, by_source, `_index`) as the target; avoid implying only raw/downloads/processed.
- **COLD_STORAGE:** No active KmiDi code should depend on COLD_STORAGE paths; extraction is surgical (ARCHIVE LAW). Confirmed: references in repo are in docs/ops (musicgen-local, transfer runs) and `final_kel/training/storage_paths.py` fallback list — all reference/documentation only.

---

## 2. Missed features

- **Emotion_Instrument_Library on SSD:** Not wired in repo. Catalog and docs point only to Google Drive. Wiring: document SSD path, add `EMOTION_INSTRUMENT_LIBRARY_ROOT` (or similar) and use it when building/loading the catalog so the SSD copy can be used when mounted.
- **SSD `audio/` vs prepare_datasets:** SSD has `audio/emotion_cremad`, `emotion_ravdess`, `emotion_tess`, `lakh_midi`, `m4singer`, `maestro`, `maestro_v3`. prepare_datasets supports emotion_*, lakh_midi, maestro (not m4singer or maestro_v3). Either add m4singer/maestro_v3 to prepare_datasets or document that `audio/` on volume is a parallel layout for manually dropped datasets.
- **musicgen-local on SSD vs workspace:** SSD `musicgen-local/ops/runs/` is missing runs present in workspace: `2026-02-13-transfer-pass-12-rulebreak-index-and-push.md`, `2026-02-13-transfer-pass-12_5-symbolic-entrypoint.md`, `2026-02-13-symbolic-entrypoint-symbolic_pass12_5.md`. Sync or document that workspace `projects/musicgen-local` is canonical for ops/runs.
- **Training gate / approval:** musicgen-local references `docs/ml/APPROVAL_CHECKLIST.md` and scripts like `check_training_gate.sh`, `open_training_gate.sh`. These exist on SSD musicgen-local; ensure workspace `projects/musicgen-local` has the same gate workflow and that any runbooks reference the repo path, not only the SSD path.
- **Backup runbooks:** SSD `backup/` contains code and runbooks (e.g. `backup/xcode/KmiDi/*.md`, `backup/.cursor/commands/*.md`, `backup/COOLIO/run_local_api.sh`, `training/run_cloud.sh`). Review for any runbook or script that should be merged into the repo (e.g. into `docs/` or `scripts/`) and deprecate duplicates.
- **KmiDi_MASTER_VAULT:** Many summary/audit docs (e.g. `CLAUDE_AGENT_GUIDE.md`, `COMPLETE_MIGRATION_STATUS.md`, `SPECTOCLOUD_IMPLEMENTATION.md`) that may inform current KmiDi; consider a one-time pass to pull actionable items into workspace docs and leave the vault as reference-only.

---

## 3. Naming / consistency

- **Sean's SSD vs KmiDi-external:** Workspace uses both. `scripts/consolidate_external_on_ssd.sh` and `docs/DATASETS_PREPARE_SCRIPT.md`, `docs/EXTERNAL_DRIVE_AUDIO_FILES.md` use **Sean's SSD**. `projects/musicgen-local/` ops runs and `docs/SSD_WORKDIR_STRUCTURE.md` use **KmiDi-external** and paths like `DevEXTERNAL`, `COLD_STORAGEEXTERNAL`. Recommendation: (1) State in one place (e.g. AGENTS.md or ENVIRONMENT.md) that the external data volume may be mounted as either "Sean's SSD" or "KmiDi-external" and that `KMIDI_DATA_ROOT` must be set to the actual mount path. (2) After running `consolidate_external_on_ssd.sh`, directory names on the volume no longer have the EXTERNAL suffix; update `docs/SSD_WORKDIR_STRUCTURE.md` and `docs/EXTERNAL_DRIVE_DATASET_SCRIPTS.md` to describe both pre- and post-consolidation layouts (or only post- if that is the standard).
- **KmiDi (volume root) vs Dev/KmiDi:** Top-level **KmiDi** on the SSD is a home-directory backup; **Dev/KmiDi** is the repo clone. Make this explicit in VOLUME_README.md on the SSD and optionally in workspace docs.
- **Single canonical prepare_datasets:** `docs/EXTERNAL_DRIVE_DATASET_SCRIPTS.md` correctly states that the active repo uses `scripts/utilities/prepare_datasets.py` and external copies are for recovery/legacy. No change; keep that policy and avoid adding scripts that duplicate prepare_datasets on the volume without pointing back to the repo.

---

## 4. Path map (code-only, tip touched)

- **_sorted:** (none in tip) — Audio_Samples, CPP_JUCE, Scripts, etc.; no code files in shallow scan.
- **audio:** (none) — Audio dirs only; ignored per scope.
- **backup:** README, .cursor/commands, COOLIO config/scripts, xcode/KmiDi .md/.py — runbooks and scripts candidate for merge.
- **build:** (empty).
- **cache:** vehicle_geometry (dir) — no code in tip.
- **`CLEANUP_RECOVERY_*`:** (logs/reports) — governance only.
- **COLD_STORAGE:** (not parsed) — doc refs only; no code dependency.
- **COOLIO:** .env.example, README, docker-compose, requirements — separate project; no KmiDi shared config in tip.
- **Datasets:** README, _index (catalog, color_coding), by_domain — aligned with docs; legacy numbered sections documented on SSD README.
- **Dev/KmiDi:** music_brain, scripts, CONSOLIDATION_SUMMARY, JUCE docs — older/different clone; many _QUARANTINE/_AUDIT only on SSD.
- **Emotion_Instrument_Library:** (no code in tip) — media only; path documented as missed feature above.
- **GH_REPOS:** kelly-listening-contract, lariat-bible, BEO-Master, etc. — scripts and READMEs; EXTERNAL_DRIVE_DATASET_SCRIPTS references GH_REPOS for legacy prepare scripts.
- **KmiDi (root):** — home backup; naming clarification recommended.
- **KmiDi_MASTER_VAULT:** Many .md, .sh, pyproject.toml, ML Kelly Training/train_mps_stub.py — reference-only; consider harvesting actionable docs.
- **ml-training-suite:** (configs, scripts, src in tip) — referenced by musicgen-local bootstrap; no extra code pulled in this pass.
- **musicgen-local:** docs/ml, schemas, scripts (gate, train, eval), ops/releases — workspace has more ops/runs; gate scripts present on both.
- **My Mac:** — user backup; not parsed for code.
- **survey autonomation:** pyproject.toml, config, tests, docs — separate project; no KmiDi dependency in tip.
- **TEST UPLOADS:** — not parsed.
- **VOLUME_README.md:** Read — describes layout; KmiDi root description should be corrected.

---

## 5. Deliverable summary

- **Improvements:** Env-based Emotion_Instrument_Library path; VOLUME_README and KmiDi-root vs Dev/KmiDi clarification; optional prepare_datasets extension for m4singer/maestro_v3; docs alignment for Datasets canonical sections and COLD_STORAGE.
- **Missed features:** Wire SSD Emotion_Instrument_Library; document or support audio/ and m4singer/maestro_v3; sync musicgen-local ops/runs; ensure training gate/approval in repo; review backup and vault for mergeable runbooks.
- **Naming/consistency:** Unify volume naming (Sean's SSD vs KmiDi-external) in docs; document post-consolidation layout (no EXTERNAL suffix); single canonical prepare_datasets already stated.

No file edits or moves were made in this phase; this report is the basis for follow-up tasks.
