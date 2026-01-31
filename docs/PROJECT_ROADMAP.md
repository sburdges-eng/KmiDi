# KmiDi MIDI Companion — Project Roadmap

**Single source of truth for project direction and next steps.** All active engineering under `~/Dev`. Stability > novelty; clarity > expansion; systems > fragments.

**Kelly brain inclusion:** The load-bearing "Brain" in this repo is the **Kelly brain** — intent → MIDI, emotion → harmony, orchestrator, music_brain, penta_core. Kelly surfaces in: `run_brain.py`, `KmiDi_CANON/brain/`, `KmiDi_CANON/body/` (KellyBrain, KellyML, kelly_ffi), `KmiDi_CANON/ui/` (kelly_brain_* commands), and `src/kelly/integrations/` (Magenta/stem-jepa). Same system; "KmiDi" = repo/product name, "Kelly" = brain/engine naming in code and UI.

**Last updated:** 2026-01-31 (Next 90 Days kickoff complete)

---

## 1. Vision & objectives

| Objective | Description |
|-----------|-------------|
| **Full reimplementation** | Replace stubs and placeholders with real implementations or explicit, documented deferrals. |
| **Spine durability** | Harden Brain → music_brain → tier1 → orchestrator and intent → process_intent → MIDI. |
| **One canonical tree** | All active code under this repo; no parallel shadow trees; forensic read-only for recovery only. |
| **Doc lifecycle** | Mark MDs FINISHED / ARCHIVE when phases or docs are complete so active work is unambiguous. |

---

## 2. Current status

| Phase | Focus | Status | Outcome |
|-------|--------|--------|---------|
| **1** | Structural durability (spine & contracts) | **Done** | CONTRACTS.md, BOOT.md, integration test, inference clean, `run_brain.py check` reports spine. |
| **2** | Full reimplementation (stubs → real or deferred) | **Done** | LLM, MCP workstation, chatbot, penta_core done; image/audio documented stub (contract satisfied). DEVELOPMENT_ROADMAP_FORENSIC.md marked ARCHIVE. |
| **3** | Direct reinforcement (tests, docs, load-bearing modules) | **Done** | Intent→MIDI test, intent_processor tests, BOOT/CONTRACTS phase order, spine checklist. |
| **4** | Validation & hardening | **Done** | Single-tree documented; stub-creep in CI; doc markers applied; refactor/recovery rules (CONTRACTS §9–10). PROJECT_ROADMAP_REIMPLEMENTATION.md marked FINISHED. |

**Spine (load-bearing path):** `run_brain.py` → `mcp_workstation.orchestrator` / `penta_core` → `music_brain.session` + `music_brain.tier1` → harmony / intent_processor / structure / groove.

**Spine checklist (reinforcement):** Before refactors, confirm: `run_brain.py`, `KmiDi_CANON/brain/mcp_workstation/orchestrator.py`, `music_brain/session/intent_processor.py`, `music_brain/tier1/midi_pipeline_wrapper.py`, `penta_core/ml/inference.py` — importable in order; `run_brain.py check` green; CONTRACTS.md and BOOT.md updated if API changes.

**Kelly brain:** Brain = Kelly brain (orchestrator, music_brain, penta_core; body: KellyBrain, KellyML; UI: kelly_brain_*; integrations: `src/kelly/`).

**References:** [CONTRACTS.md](CONTRACTS.md), [BOOT.md](BOOT.md), [DATA_AND_TRAINING.md](DATA_AND_TRAINING.md).

---

## 3. Phase summary & remaining work

### Phase 1 — Structural durability ✅

- [x] Contracts in `docs/CONTRACTS.md` (Intent→MIDI, LLM→Intent, Orchestrator, Boot).
- [x] Integration test: intent → process_intent → MIDI (no LLM).
- [x] `penta_core/ml/inference.py` conflict resolved.
- [x] `run_brain.py check` lists music_brain, mcp_workstation, penta_core, tier1.

### Phase 2 — Full reimplementation 🔄

| Domain | Location | Status | Next |
|--------|----------|--------|------|
| LLM → Intent | `mcp_workstation/llm_reasoning_engine.py` | Done | Rule-based parse + optional llama-cpp; image/audio prompts template-based. |
| Image generation | `mcp_workstation/image_generation_engine.py` | Documented stub | Contract satisfied; status stubbed/completed/failed; model path/env in BOOT. Real pipeline optional. |
| Audio generation | `mcp_workstation/audio_generation_engine.py` | Documented stub | Contract satisfied; stubbed vs failed vs completed in return; deps in BOOT. |
| MCP workstation | `phases.py`, `proposals.py`, `models.py` | Done | Phases/proposals in-memory; orchestrator dashboard/task tracking. |
| music_brain chatbot | `music_brain/chatbot/agent.py` | Deferred | Documented deferral; [deferred] prefix on responses. |
| penta_core | `penta_core/ml/training/augmentation.py` | Done | Implementation complete; doc updated. |
| music_brain realtime | `music_brain/realtime/events.py` | Minimal stub | Keep stub; document "full impl requires Logic bridge". |
| Vocal (prrot/Parrot) | `body/prrot/`, `music_brain/vocal/parrot.py`, `voice/cpp_bridge.py` | Complete | prrot = C++ phoneme/voice engine; Parrot = Python voice learning/synthesis; bridge = OSC Python→C++. |

**Rule:** No new stubs without a short doc comment and ticket; preserve orchestrator API.

### Phase 3 — Direct reinforcement ✅ (minor open)

- [x] Integration test: `CompleteSongIntent.from_flat` → process_intent → MIDI.
- [x] Unit tests for intent_processor (e.g. HARMONY_ModalInterchange, default key/mode).
- [x] Phase order and stub vs failed semantics in BOOT/CONTRACTS.
- [x] Optional: spine reinforcement checklist — see **Spine checklist** below.
- [x] When Phase 3 fully closed: mark Phase 3 checklist MD FINISHED per §9 in reimplementation roadmap.

### Phase 4 — Validation & hardening ✅

- [x] **Single tree:** All active code under this repo; forensic read-only; no parallel clone (documented in roadmap and governance).
- [x] **No stub creep:** CI runs `scripts/check_stub_creep.py --allow-docs`; undocumented stubs fixed or documented (BOOT.md, .github/workflows/ci.yml).
- [x] **Boot check:** `run_brain.py check` passes.
- [x] **Integration:** Intent → process_intent → MIDI test green.
- [x] **Doc markers:** DEVELOPMENT_ROADMAP_FORENSIC.md ARCHIVE; CONTRACTS.md FINISHED; PROJECT_ROADMAP_REIMPLEMENTATION.md FINISHED per §9.
- [x] **Hardening in repo:** Checkpoints/models — [DATA_AND_TRAINING.md](DATA_AND_TRAINING.md); ref in [CONTRACTS.md](CONTRACTS.md) §8.
- [x] **Refactor law:** Documented in [CONTRACTS.md](CONTRACTS.md) §9 (spine file > ~400 lines → split by responsibility, single public API).
- [x] **Recovery rule:** Documented in [CONTRACTS.md](CONTRACTS.md) §10 (no recoverable code path → reference roadmap / FORENSIC_RECOVERY_REPORT / recovery-code-path.mdc).
- [x] When Phase 4 complete: completion banner added to PROJECT_ROADMAP_REIMPLEMENTATION.md per §9.

---

## 4. Next 90 days (prioritized)

**Started:** 2026-01-31. Roadmap execution complete (Phases 1–4). **Next 90 Days kickoff (§4.2, §4.3) complete:** 2026-01-31. Current focus: **maintenance and incremental improvement**.

### 4.1 Priorities

1. **Optional: Image/audio real pipelines**  
   Image/audio engines are documented stubs (contract satisfied). When models are in scope: load real SD/audiocraft pipeline; keep orchestrator API.

2. **Governance (TODO.md)**  
   Confirm datasets under `~/Datasets` and checkpoints under `~/Models`; Magenta/stem-jepa documented in `src/kelly/integrations/README.md`; pre-push hook: `cp scripts/pre-push-hook.sh .git/hooks/pre-push && chmod +x .git/hooks/pre-push`.

3. **Doc lifecycle**  
   When adding new plan/checklist MDs, add them to PROJECT_ROADMAP_REIMPLEMENTATION.md §9 and assign completion marker.

### 4.2 First steps (kickoff)

- [x] **One-time path check:** Run `./scripts/verify_data_paths.sh` (or verify manually); create `~/Datasets` and `~/Models` if doing training. See [DATA_AND_TRAINING.md](DATA_AND_TRAINING.md). No large data in repo. *(Completed 2026-01-31: all paths present.)*
- [x] **Pre-push hook (optional):** Install for local sanity before push: `cp scripts/pre-push-hook.sh .git/hooks/pre-push && chmod +x .git/hooks/pre-push` (CI already runs check + stub-creep on push/PR). *(Completed 2026-01-31.)*
- [x] **TODO.md:** Work through unchecked governance items in [TODO.md](../TODO.md) (boot, data paths, experiments policy, housekeeping). *(Completed 2026-01-31: optional imports documented, experiments promotion policy added, integration bridge opportunities identified.)*
- [ ] **Image/audio (when in scope):** Leave as documented stub until models/env are ready; then wire real pipeline per BOOT optional model paths.

### 4.3 Issues list steps ([ISSUES_LIST.md](ISSUES_LIST.md))

- [x] **Before recreating code:** Search [GIT_RESTORE_PATHWAYS.md](GIT_RESTORE_PATHWAYS.md), `git log -S "symbol"`, `docs/.index/symbol_index_canon.tsv`. *(Completed 2026-01-31: workflow documented in CONTRACTS.md §10.)*
- [x] **Incomplete modules:** See [INCOMPLETE_MODULES_LAST_KNOWN_PATHS.md](INCOMPLETE_MODULES_LAST_KNOWN_PATHS.md) when restoring or implementing stubbed modules (Spectocloud, api_server lyrics/interrogate, etc.). *(Completed 2026-01-31: Spectocloud marked complete.)*
- [x] **Stub creep:** Fix or convert to documented deferral; update `scripts/check_stub_creep.py` ALLOWED_CONTEXTS if intentional. Current hits: api_server (interrogate, humanizer, lyrics) — see [ISSUES_LIST.md](ISSUES_LIST.md) §1. *(Completed 2026-01-31: all 4 stubs resolved with [deferred] documentation.)*

---

## 5. Governance alignment

| Law | Doc / location | Status |
|-----|----------------|--------|
| **PRIME** | All active work in `~/Dev` | ✅ |
| **BOOT** | [BOOT.md](BOOT.md), `run_brain.py` | ✅ |
| **DATA** | [DATA_AND_TRAINING.md](DATA_AND_TRAINING.md), `~/Datasets`, `~/Models` | Documented |
| **ENV** | Envs outside repo (e.g. `~/envs`, micromamba); see ENV_AND_TMUX | — |
| **EXPERIMENT** | `experiments/exp_NNN_description`; [experiments/README.md](../experiments/README.md) | ✅ |
| **TRAINING** | Configs, manifests, checkpoints outside repo; [DATA_AND_TRAINING.md](DATA_AND_TRAINING.md) | ✅ |
| **BRIDGE** | ML↔DSP, Python↔C++; `src/kelly/integrations/` | Document scope |
| **REFACTOR** | Spine file size / split rule in Phase 4 | Pending |

---

## 6. Related docs

| Doc | Purpose |
|-----|--------|
| [PROJECT_ROADMAP_REIMPLEMENTATION.md](PROJECT_ROADMAP_REIMPLEMENTATION.md) | Full phase breakdown, contracts list, reimplementation table, §9 doc markers. |
| [ISSUES_LIST.md](ISSUES_LIST.md) | Current env status, stub creep, incomplete modules, historical issues, recovery and next steps (aligned with §4.3). |
| [DEVELOPMENT_ROADMAP_FORENSIC.md](DEVELOPMENT_ROADMAP_FORENSIC.md) | **ARCHIVE.** Historical DAiW/Music-Brain queue (CLI, MCP 22 tools, audio); preserved for audit/recovery. |
| [CONTRACTS.md](CONTRACTS.md) | Intent→MIDI, LLM→Intent, Orchestrator, Boot contracts. |
| [BOOT.md](BOOT.md) | Modes, dependency order, check list, loop, stub vs failed semantics. |
| [DATA_AND_TRAINING.md](DATA_AND_TRAINING.md) | Datasets, checkpoints, experiments, run manifest, training safety. |
| [TODO.md](../TODO.md) | Governance-aligned housekeeping and integration tasks. |

---

*Stability > novelty. Clarity > expansion. Systems > fragments.*
