# Git restore pathways — LLM & engine intent

**Purpose:** Stop agents from recreating LLM and engine intent code by documenting where original source lives and how to restore or diff from git history.

**References:** `FORENSIC_RECOVERY_REPORT.md`, `docs/DEVELOPMENT_ROADMAP_FORENSIC.md`, `docs/PROJECT_ROADMAP_REIMPLEMENTATION.md`

---

## 1. Where the code actually lives

### 1.1 Intent schema & processor (session)

| What | Canonical git source | In KmiDi canon |
|------|----------------------|----------------|
| **intent_schema.py** (CompleteSongIntent, SongRoot, SongIntent, TechnicalConstraints, SystemDirective, from_dict, from_flat) | **Forensic DAiW-Music-Brain** repo | `KmiDi_CANON/brain/music_brain/session/intent_schema.py` (on disk) |
| **intent_processor.py** (process_intent, rule-breaking, chord/groove/arrangement generation) | **Forensic DAiW-Music-Brain** repo | `KmiDi_CANON/brain/music_brain/session/intent_processor.py` (on disk) |
| **generator.py**, **interrogator.py**, **teaching.py**, **therapy_prompts.py** | **Forensic DAiW-Music-Brain** repo | `KmiDi_CANON/brain/music_brain/session/*` (on disk) |

- **Forensic repo path:** `~/Dev/_FORENSIC_READONLY_KMIDI/iDAWComp/DAiW-Music-Brain`  
  (Read-only; do not develop there. Extract only.)

- **KmiDi repo:** `~/Dev/KmiDi MIDI Companion`  
  **Critical:** `KmiDi_CANON/brain/music_brain/` is **untracked** in this repo. It was restored from forensic (Phase 6) but never committed. Commit it to avoid re-recreation and to make git history searches find it.

### 1.2 LLM reasoning engine

| What | Git history | In KmiDi canon |
|------|-------------|----------------|
| **llm_reasoning_engine.py** (parse_user_intent → CompleteSongIntent, rule-based + optional GGUF LLM) | **None in either repo.** File was recreated by agents; no historical version exists in git. | `KmiDi_CANON/brain/mcp_workstation/llm_reasoning_engine.py` (untracked) |

- **Pathway:** The current `llm_reasoning_engine.py` correctly imports `music_brain.session.intent_schema` (CompleteSongIntent, SongRoot, SongIntent, TechnicalConstraints, SystemDirective) and uses `from_dict` / rule-based parse. **Do not recreate it.** Commit it so it enters history.

### 1.3 C++ engine / intent (body)

| What | Git source |
|------|------------|
| **IntentPipeline**, **RuleBreakEngine**, **IntentFrame**, **EngineContract**, **intent_processor** (C++) | **KmiDi MIDI Companion** repo, commit `27148e6f` and current tree. Paths: `KmiDi_CANON/body/engine/`, `body/core/intent_ir/`, `body/core/intent_processor.*` |

- These **are** in KmiDi git history. Search with:  
  `git log --all --oneline -- "KmiDi_CANON/body/engine/*" "KmiDi_CANON/body/core/intent*"`

---

## 2. Forensic DAiW-Music-Brain — branches and key commits

Use this repo only for **read-only** restore or diff. Do not develop there.

```text
Repo:  ~/Dev/_FORENSIC_READONLY_KMIDI/iDAWComp/DAiW-Music-Brain
Branches (examples): main, origin/main, claude/*, copilot/*
```

### Key commits for session/intent (from `git log --all -- music_brain/session/`)

| Commit (short) | Description |
|---------------|-------------|
| **b3cd0eb** | Extract project files and comprehensive CLAUDE.md (initial music_brain/session) |
| **c5b3450** | Extract and add DAiW Music Brain codebase for test coverage (generator, intent_processor, intent_schema) |
| **dc58aac** | Sync expanded intent_schema.py to main package |
| **230c8a8** | Add HAPPY emotion to intent schema and emotion taxonomy |
| **0391216** | Implement Phase 2: Complete all rule-breaking processors (intent_processor, intent_schema) |
| **0b23796** | Fix debug issues: intent_processor, intent_schema |
| **317bab9** | Refactor therapy session scale tests; intent_schema |

### Restore a single file from forensic (no checkout)

From the **KmiDi** workspace (or any dir), to dump a file from forensic at a given commit:

```bash
FORENSIC="/Users/seanburdges/Dev/_FORENSIC_READONLY_KMIDI/iDAWComp/DAiW-Music-Brain"

# Intent processor at Phase 2 implementation
git -C "$FORENSIC" show 0391216:music_brain/session/intent_processor.py > /tmp/intent_processor_0391216.py

# Intent schema at “expanded” sync
git -C "$FORENSIC" show dc58aac:music_brain/session/intent_schema.py > /tmp/intent_schema_dc58aac.py

# List session files at a commit
git -C "$FORENSIC" ls-tree -r --name-only 0391216 -- music_brain/session/
```

Then diff or copy selectively into `KmiDi MIDI Companion/KmiDi_CANON/brain/music_brain/session/`.

### Search forensic history for content (pickaxe)

```bash
cd "/Users/seanburdges/Dev/_FORENSIC_READONLY_KMIDI/iDAWComp/DAiW-Music-Brain"
git log -p --all -S "process_intent" -- "*.py"
git log -p --all -S "CompleteSongIntent" -- "*.py"
git log -p --all -S "from_dict" -- music_brain/session/
```

---

## 3. KmiDi MIDI Companion — what’s in git vs on disk

| Path | In git? | Action |
|------|---------|--------|
| `KmiDi_CANON/brain/music_brain/` | **No** (untracked) | **Commit** so session, structure, realtime, etc. are in history and searchable. |
| `KmiDi_CANON/brain/mcp_workstation/llm_reasoning_engine.py` | **No** (untracked) | **Commit** so agents don’t recreate it. |
| `KmiDi_CANON/brain/mcp_workstation/orchestrator.py` | Yes (from 27148e6f) | Already in history. |
| `KmiDi_CANON/brain/mcp_workstation/audio_generation_engine.py` | Yes (from 27148e6f) | Already in history. |
| `KmiDi_CANON/brain/penta_core/ml/*` | Yes (from 27148e6f) | Already in history. |
| `KmiDi_CANON/brain/kmidi_gui/*` | Yes (from 27148e6f) | Already in history. |
| `KmiDi_CANON/body/engine/*`, `body/core/intent_ir/*` | Yes | Search with `git log --all -- <path>`. |

Commit 27148e6f (“KmiDi Companion dev: working tree after forensic recovery and cold storage”) added kmidi_gui, mcp_workstation (orchestrator, audio_generation_engine), penta_core/ml. It did **not** add `brain/music_brain/` or `llm_reasoning_engine.py`; those exist only on disk.

---

## 4. Recommended next steps

1. **Commit untracked brain code in KmiDi** so history searches find it:
   - `KmiDi_CANON/brain/music_brain/`
   - `KmiDi_CANON/brain/mcp_workstation/llm_reasoning_engine.py`
   - Other untracked mcp_workstation modules if they are part of the spine (e.g. cognitive_router, models, ai_specializations as needed).

2. **Before “recreating” LLM or intent:**
   - Search this doc and `FORENSIC_RECOVERY_REPORT.md`.
   - Run `git log --all --oneline -- "**/llm*" "**/intent*" "**/session/*"` in both repos.
   - Restore from forensic with `git -C <forensic> show <commit>:path` and diff into canon.

3. **Treat forensic as quarry:** Extract specific files or functions when needed; do not promote forensic to active workspace (per FORENSIC_RECOVERY_REPORT Phase 5–6).

---

*Last updated: 2026-01-31. After committing music_brain and llm_reasoning_engine, re-run the git log commands above and update this doc if paths or commits change.*
