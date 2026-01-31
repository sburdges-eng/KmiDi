# KmiDi Project Roadmap — Full Reimplementation, Reinforcement & Structural Durability

**Purpose:** Full reimplementation of functions, direct focused reinforcement of critical paths, and durable structural discipline so the canon remains the single source of truth and scales without fragmentation.

**References:** `FORENSIC_RECOVERY_REPORT.md`, `docs/BOOT.md`, `docs/DEVELOPMENT_ROADMAP_FORENSIC.md`, `docs/GIT_RESTORE_PATHWAYS.md`, `docs/FUNCTION_INDEX_README.md` (function/path index), `.cursor/rules/recovery-code-path.mdc`

---

## 1. Objectives

| Objective | Description |
|-----------|-------------|
| **Full reimplementation** | Replace all stubs and placeholder code with real implementations; no "restore from forensic if needed" left unresolved. |
| **Direct focused reinforcement** | Strengthen load-bearing spine (Brain → music_brain → tier1 → orchestrator) and critical paths (intent → process_intent → MIDI; LLM → intent; penta_core inference). |
| **Durability of structure** | Enforce clear contracts, tests, and docs so the structure survives refactors and stays one canonical tree. |
| **Doc lifecycle** | Mark MDs as FINISHED / DO NOT READ / ARCHIVE when a phase or doc is completed so active work is never confused with historical or obsolete content. |

---

## 2. Current State (Baseline)

- **Canon:** `KmiDi MIDI Companion` / `KmiDi_CANON` — single active tree (body = JUCE, brain = Python, ui = Tauri).
- **music_brain:** Restored from forensic; `process_intent` feeds tier1 MIDI path; `CompleteSongIntent` (nested + workflow fields) and `from_flat` in place.
- **Stubs / gaps still present:**
  - **LLM reasoning:** `mcp_workstation/llm_reasoning_engine.py` — stub; returns minimal `CompleteSongIntent`; needs real parse/generate from model.
  - **Image / audio engines:** `image_generation_engine.py`, `audio_generation_engine.py` — stub or fallback behavior when models missing.
  - **MCP workstation:** `debug.py`, `cpp_planner.py`, `phases.py`, `proposals.py`, `ai_specializations.py`, `models.py` — stubs for orchestrator.
  - **music_brain:** `chatbot/agent.py` (echo stub), `realtime/events.py` (minimal stub without Logic deps).
  - **penta_core:** `inference.py` — resolve merge conflict; `training/augmentation.py` — stub logic.
- **Critical merge conflict:** `penta_core/ml/inference.py` — resolve before production use.
- **Doc state:** No completion markers yet; all listed MDs are active. Upon phase completion, apply markers per §9.

---

## 3. Phase 1 — Structural Durability (Spine & Contracts)

**Goal:** Harden the load-bearing spine and define durable contracts so reimplementation doesn’t fragment the system.

### 3.1 Spine definition

- **Entry:** `run_brain.py` (see `docs/BOOT.md`).
- **Spine:** `run_brain` → `mcp_workstation.orchestrator` / `penta_core` → `music_brain.session` + `music_brain.tier1` → `music_brain.harmony` / `intent_processor` / structure / groove.
- **Rule:** No new top-level entry points; no parallel "shadow" trees. All active code under `KmiDi_CANON`.
- **Detail:** Spine modules must remain importable in dependency order (penta_core, music_brain.session, music_brain.tier1, mcp_workstation); `run_brain.py check` validates this order.

### 3.2 Contracts to document and enforce

| Contract | Owner | Description |
|----------|--------|-------------|
| **Intent → MIDI** | `music_brain.tier1.midi_pipeline_wrapper` | `CompleteSongIntent` → `process_intent` → harmony → `generate_midi_from_harmony` → MIDI file; return dict with `status`, `midi_path`, `chords`, `rule_broken`, `groove_tempo`. |
| **LLM → Intent** | `mcp_workstation.llm_reasoning_engine` | User text → `CompleteSongIntent` (nested: song_root, song_intent, technical_constraints); optional image/audio prompts. |
| **Orchestrator workflow** | `mcp_workstation.orchestrator` | Phases: LLM intent → from_flat if flat else use forensic intent → MIDI pipeline → optional image/audio; single `CompleteSongIntent` in/out. |
| **Boot** | `run_brain.py` | Modes: penta, orchestrator, gui (stub), check; dependency order per BOOT.md. |

- **Detail:** Each contract has a single owner file; changes to input/output shape require updating CONTRACTS.md and callers together.

### 3.3 Tasks

- [x] Document the four contracts above in `docs/CONTRACTS.md` (or extend BOOT.md).
- [x] Add a minimal integration test: intent → process_intent → MIDI path (no LLM); run in CI or pre-commit.
- [x] Resolve merge conflict in `penta_core/ml/inference.py` (accept full implementation; remove conflict markers). *(Already clean.)*
- [x] Ensure `run_brain.py check` reports music_brain, mcp_workstation, penta_core, tier1; document in BOOT.
- [x] Add `docs/CONTRACTS.md` to the list of MDs that receive a completion marker when Phase 1 is done (see §9).

**Success:** Spine documented; one integration test green; inference.py clean; check mode accurate.

---

## 4. Phase 2 — Full Reimplementation of Functions

**Goal:** Replace every stub and "restore from forensic if needed" with a real implementation or a deliberate, documented deferral.

### 2.1 By domain (priority order)

| Domain | Location | Current | Target |
|--------|----------|---------|--------|
| **LLM → Intent** | `mcp_workstation/llm_reasoning_engine.py` | Stub returns minimal intent | Load model; parse user text → nested `CompleteSongIntent`; generate image/audio prompts; keep API (parse_user_intent, generate_image_prompts, etc.). |
| **Intent schema / workflow** | `music_brain.session.intent_schema` | Done (nested + from_flat + workflow fields) | No change; treat as reference for other reimpl. |
| **MIDI path** | `music_brain.tier1.midi_pipeline_wrapper` | Done (process_intent → harmony → MIDI) | Optional: add groove humanization from process_intent groove; document. |
| **Image generation** | `mcp_workstation/image_generation_engine.py` | Stub / no-op load | Load SD (or chosen) pipeline; generate from prompt; return asset path or bytes; keep orchestrator API. |
| **Audio generation** | `mcp_workstation/audio_generation_engine.py` | Stubbed when audiocraft missing | Document deps; harden fallback; ensure orchestrator gets clear status (stubbed vs failed vs completed). |
| **MCP workstation** | `phases.py`, `proposals.py`, `models.py`, `cpp_planner.py`, `ai_specializations.py`, `debug.py` | Stubs | Implement or narrow scope: at least phases/proposals/models sufficient for orchestrator dashboard and task tracking; cpp_planner/ai_specializations can remain minimal until C++/AI workflows are in scope. |
| **music_brain chatbot** | `music_brain/chatbot/agent.py` | Echo stub | Real agent or explicit "deferred" note; no silent stub in user-facing path. |
| **music_brain realtime** | `music_brain/realtime/events.py` | Minimal stub without Logic | Keep stub for import safety; document "full impl requires Logic bridge"; no duplicate realtime spine. |
| **penta_core** | `penta_core/ml/inference.py`, `training/augmentation.py` | Conflict + stub | Resolve inference; replace augmentation stub with real logic or document. |

- **Detail:** For each domain, add a one-line "completion note" in the doc header when reimpl is done (e.g. "LLM reimpl completed YYYY-MM-DD; see CONTRACTS.").
- **Detail:** DEVELOPMENT_ROADMAP_FORENSIC.md is historical; when Phase 2 is complete, mark it ARCHIVE per §9 so it is not read as active plan.

### 2.2 Reimplementation rules

- **Reference:** When no local path exists, use `FORENSIC_RECOVERY_REPORT.md`, `_FORENSIC_READONLY_KMIDI` (surgical read-only), or main/master (see `.cursor/rules/recovery-code-path.mdc`).
- **No new stubs:** New code either implements behavior or is explicitly deferred with a short doc comment and a ticket/issue.
- **Orchestrator API:** Preserve `execute_workflow(user_intent_text, …) → CompleteSongIntent` and existing resource locking; reimplement behind existing entry points.
- **Detail:** Any new env var or model path (e.g. LLM, SD, audiocraft) must be documented in ENV_AND_TMUX or BOOT and defaulted so check mode still runs without them.

### 2.3 Tasks (concise)

- [ ] LLM reasoning engine: real model load + parse → `CompleteSongIntent` (nested) + image/audio prompt generation.
- [ ] Image engine: real pipeline load + generate; document model path/env.
- [ ] Audio engine: document deps; distinguish stubbed vs failed vs completed in return.
- [ ] MCP workstation: implement phases/proposals/models enough for orchestrator; document scope for cpp_planner/ai_specializations.
- [ ] music_brain chatbot: real agent or documented deferral.
- [ ] penta_core: resolve inference merge; replace or document augmentation stub.
- [ ] When Phase 2 is complete, mark DEVELOPMENT_ROADMAP_FORENSIC.md as ARCHIVE (see §9); do not delete.

**Success:** No remaining "restore from forensic if needed" in active code paths; each stub either implemented or explicitly deferred in docs.

---

## 5. Phase 3 — Direct Focused Reinforcement

**Goal:** Reinforce the critical paths and load-bearing modules so they are testable, documented, and resilient.

### 3.1 Critical paths

| Path | Reinforcement |
|------|----------------|
| **User text → MIDI** | LLM (reimpl) → intent → from_flat/or use forensic intent → process_intent → tier1 MIDI; one integration test covering intent → MIDI without LLM; one optional test with LLM stub returning nested intent. |
| **Intent → process_intent** | process_intent always receives nested `CompleteSongIntent`; default key/mode in tier1 when missing; no silent fallback to empty harmony. |
| **Orchestrator resources** | Keep locks for llm, midi_gen, image_gen, audio_gen; document timeout and restart behavior (see BOOT.md --loop). |
| **Boot** | `run_brain.py check` lists all spine modules; penta and orchestrator modes start without ad-hoc path hacks. |
- **Detail:** Logging on the critical path (intent received, process_intent called, MIDI written) must be consistent so debugging does not require reading code.

### 3.2 Load-bearing modules (reinforce first)

1. **music_brain.session.intent_schema** — `CompleteSongIntent`, `from_flat`, workflow fields; already durable; keep to_dict/from_dict/save/load in sync.
2. **music_brain.session.intent_processor** — `process_intent`; already used by tier1; add a few unit tests for key/mode and rule_break.
3. **music_brain.tier1.midi_pipeline_wrapper** — Single place that turns intent into MIDI; keep process_intent → harmony → generate_midi_from_harmony; add groove to MIDI in a later iteration if needed.
4. **mcp_workstation.orchestrator** — Single workflow entry; keep from_flat vs forensic-intent handling; document phase order and error handling.
5. **penta_core/ml/inference** — After conflict resolve; ensure inference API is stable and used only via defined entry points.
6. **run_brain.py** — Load-bearing entry; keep mode dispatch and check logic minimal; any new mode must be listed in BOOT and in the completion-marker list if it gets its own MD.

### 3.3 Tasks

- [x] Add integration test: `CompleteSongIntent.from_flat(...)` → `process_intent` → `MIDIGenerationPipeline.generate_midi` → assert file exists and status completed.
- [x] Add unit tests for intent_processor (e.g. HARMONY_ModalInterchange, default key/mode).
- [x] Document in BOOT or CONTRACTS: phase order (LLM → intent → MIDI → image → audio), error handling, and stub vs failed semantics.
- [ ] Optional: add a small "reinforcement" checklist in this doc or in CODEOWNERS for spine files.
- [ ] Add BOOT.md and CONTRACTS.md to the completion-marker list; when Phase 3 is done, mark any "phase 3 checklist" MD as FINISHED per §9.

**Success:** Critical path tested; load-bearing modules listed and documented; BOOT/CONTRACTS updated.

---

## 6. Phase 4 — Validation & Hardening

**Goal:** Ensure the structure stays durable over time: no sprawl, no duplicate spines, clear ownership.

### 4.1 Validation

- [ ] **Single tree:** All active code under `KmiDi MIDI Companion`; no parallel clone for "new" work; forensic read-only and used only for restore reference.
- [ ] **No stub creep:** CI or pre-commit grep for "stub" / "restore from forensic if needed" in active code; fix or convert to documented deferral.
- [x] **Boot check:** `run_brain.py check` passes and lists music_brain, mcp_workstation, penta_core, tier1.
- [x] **Integration:** At least one test that runs intent → process_intent → MIDI and asserts success.
- [ ] **Doc markers:** All MDs that were "active" for a phase are marked FINISHED or ARCHIVE per §9 so CI/docs scripts can exclude "do not read" content if desired.

### 4.2 Hardening

- [ ] Document in this repo: where checkpoints/models live (`~/Models`, `~/Datasets` per governance); no large outputs under repo.
- [ ] Refactor law: when a spine file grows beyond a threshold (e.g. 400 lines), split by responsibility and keep a single public API for the orchestrator.
- [ ] Recovery rule: any "no recoverable code path" must reference this roadmap or FORENSIC_RECOVERY_REPORT or main/master (see recovery-code-path.mdc).
- [ ] When Phase 4 is complete, add this roadmap to the "FINISHED" list in §9 and prepend the doc with the standard completion banner so it is clear the plan is executed.

**Success:** Validation automated or scripted; hardening rules in docs; recovery rule applied.

---

## 7. Summary Table

| Phase | Focus | Outcome |
|-------|--------|---------|
| **1** | Structural durability | Spine documented; contracts in CONTRACTS/BOOT; inference conflict resolved; one integration test. |
| **2** | Full reimplementation | Stubs replaced or explicitly deferred; LLM, image, audio, MCP, chatbot, penta_core addressed. |
| **3** | Direct reinforcement | Critical path tested; load-bearing modules listed and documented; BOOT/CONTRACTS updated. |
| **4** | Validation & hardening | Single tree enforced; no stub creep; boot check and integration test green; refactor/recovery rules in docs. |
| **Doc markers** | §9 | Every completed phase or superseded MD gets a completion marker (FINISHED / DO NOT READ / ARCHIVE) so active work is unambiguous. |

---

## 8. Maintenance

- **Owner:** KmiDi maintainers; keep this roadmap under `docs/` and link from README or BOOT.
- **Updates:** When a stub is replaced or a contract changes, update this doc and CONTRACTS/BOOT.
- **References:** FORENSIC_RECOVERY_REPORT.md, docs/BOOT.md, docs/DEVELOPMENT_ROADMAP_FORENSIC.md, .cursor/rules/recovery-code-path.mdc.

---

## 9. Doc completion markers (MDs to mark on completion)

Upon completion of a phase or when a doc is superseded, **mark the MD so readers know not to treat it as active work**. Use one of the following banners at the **top** of the file (after any frontmatter).

### 9.1 Banner format

Place at the very top of the Markdown file:

```markdown
<!-- STATUS: FINISHED | DO NOT READ | ARCHIVE -->
<!-- Completed: YYYY-MM-DD | Reason: e.g. Phase 1 complete, superseded by CONTRACTS.md -->
```

Or as a short visible block:

```markdown
> **STATUS: FINISHED** — Completed YYYY-MM-DD. Do not use as active checklist. See CONTRACTS.md / PROJECT_ROADMAP_REIMPLEMENTATION.md for current state.
```

### 9.2 When to use each marker

| Marker | When to use |
|--------|-------------|
| **FINISHED** | Phase or doc is complete; tasks done; content is reference only, not an active todo list. |
| **DO NOT READ** | Doc is obsolete or misleading for current work; kept for history only; do not follow. |
| **ARCHIVE** | Doc is historical (e.g. forensic roadmap); preserved for audit/recovery; not the active plan. |

### 9.3 MDs that must be marked upon completion

| MD | Mark when | Marker |
|----|-----------|--------|
| `docs/CONTRACTS.md` | Phase 1 complete (contracts written and stable) | FINISHED |
| `docs/BOOT.md` | No separate "completion" — keep active; add a "Last verified" date when check list is updated. | (optional: "Last verified YYYY-MM-DD") |
| `docs/DEVELOPMENT_ROADMAP_FORENSIC.md` | Phase 2 complete (reimpl done; forensic plan no longer the active roadmap) | ARCHIVE |
| `docs/PROJECT_ROADMAP_REIMPLEMENTATION.md` | Phase 4 complete (full roadmap executed) | FINISHED |
| Any phase-specific checklist MD (e.g. `docs/PHASE1_CHECKLIST.md`) | When that phase is complete | FINISHED |
| `FORENSIC_RECOVERY_REPORT.md` (repo root) | Already historical; optional: mark ARCHIVE when canonical tree is stable. | ARCHIVE (optional) |

### 9.4 Rules

- **Do not delete** marked MDs; keep them for audit and recovery (see recovery-code-path.mdc).
- **Search:** Grep for `STATUS: FINISHED`, `DO NOT READ`, `ARCHIVE` to list completed/historical docs.
- **New docs:** When adding a new plan or checklist MD, add it to this table and assign a marker for when it will be completed.
