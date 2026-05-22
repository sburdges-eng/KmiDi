# KmiDi_FINAL + KmiDi merge plan (worktree / single repo)

**Goal:** Merge KmiDi_FINAL (and any nested KmiDi) into the main KmiDi monorepo as a unified worktree: bring in **PRROT**, engine/DSP, and other non-UI features; **exclude** UI implementations (React components, plugin UIs, standalone app UIs) so the canonical UI remains Tauri + React in this repo.

---

## 1. What to include (from KmiDi_FINAL)

| Area | Path (in KmiDi_FINAL) | Action |
|------|------------------------|--------|
| **PRROT** (memory safety / realtime) | `python/prrot/`, engine PRROT usage | Copy or subtree-merge into `music_brain/` or `engine/`; wire into CMake/build. |
| **Engine / DSP** | `engine/src/dsp`, `engine/src/voice`, `engine/cpp_music_brain` (non-UI) | Merge into repo `engine/`, `include/`, `src/` (or `src_penta-core`). Prefer same C ABI and rt_harness compatibility. |
| **Intent IR v1** (if not already in main) | Intent IR build, schema, plugin processor updates | Main repo already has `engine/intent_ir/`; align schema and any C++ Intent IR usage. |
| **Music Brain / learning** | `python/music_brain/learning/`, `voice/`, pipeline docs | Merge into `music_brain/`; keep one API surface (`music_brain.api`). |
| **Config / models** | Training configs, model references (no large binaries) | Copy into `config/`, `experiments/` as needed. |
| **Scripts** | Build, robustness, verification scripts | Merge into `scripts/`; avoid duplicate entrypoints. |

---

## 2. What to exclude (UI implementations)

| Area | Path (in KmiDi_FINAL) | Reason |
|------|------------------------|--------|
| Standalone / macOS app UI | `apps/macOS/`, AppKit/legacy UI | Canonical desktop is Tauri + React (ADR 001). |
| React component library | Any KmiDi_FINAL-specific React that duplicates or diverges from `src/` | Single UI path: this repo’s `src/`. |
| Plugin editor UI | JUCE editor, custom plugin GUIs | Keep plugin host/processing; exclude or replace with minimal/headless plugin UI if needed. |
| Qt / JUCE UI surfaces | Qt6 UI, JUCE UI for desktop | Already off in main (AGENTS: BUILD_DESKTOP=OFF, KMIDI_BUILD_JUCE_UI=OFF by default). |

---

## 3. Concrete merge strategy

**Option A — In-repo copy (KmiDi_FINAL already at `KmiDi_FINAL/`)**  
1. Use `KmiDi_FINAL/` as source (no separate worktree). Run from repo root:
   ```bash
   bash scripts/merge-from-kmidi-final.sh
   ```
2. The script copies only chosen dirs:
   - **C++:** `engine/include/prrot` → `include/prrot`; `engine/include/penta` → `include/penta`; `engine/src_penta-core/*` → `src_penta-core/`; `engine/src/dsp` → `engine/src/dsp`; `engine/intent_ir` → `engine/intent_ir`.
   - **Python:** `python/prrot` → `music_brain/prrot`; `python/music_brain/learning` → `music_brain/learning`; `python/music_brain/voice` → `music_brain/voice`.
   - **Config / scripts:** selected YAML from `configs/` → `config/`; robustness/verify scripts → `scripts/`.
3. Resolve includes, CMake targets, and Python imports in the main repo (see §4).  
4. Run main repo build and tests; do not commit KmiDi_FINAL UI or duplicate entrypoints.

**Option A (external worktree)** — If KmiDi_FINAL is a separate repo:  
1. `git worktree add ../KmiDi_FINAL_wt KmiDi_FINAL/main` (or add as remote and checkout).  
2. Run the same copy logic from that worktree into this repo (or set `FINAL=../KmiDi_FINAL_wt` in the script).  
3. Same steps 3–4 as above.

**Option B — Subtree merge (single history)**  
1. Add KmiDi_FINAL as a remote: `git remote add kmidi-final <url>`.  
2. `git subtree add --prefix=kmidi-final-incoming kmidi-final main --squash` (or no squash).  
3. Move only the desired subpaths from `kmidi-final-incoming/` into the canonical layout (e.g. `engine/`, `music_brain/`); delete the rest (including UI).  
4. Remove `kmidi-final-incoming/` and rely on one layout.

**Option C — Copy only (no shared history)**  
1. Manually copy PRROT, engine/DSP, and music_brain additions from KmiDi_FINAL into this repo.  
2. Update CMakeLists.txt and pyproject.toml; add any new deps.  
3. Document origin in a single `docs/KMIDI_FINAL_MERGED.md` (list of merged paths and date).

---

## 4. Suggested order of operations (after running merge script)

1. **CMake / C++**  
   - Wire `include/prrot` and any new sources into root `CMakeLists.txt` or `engine/CMakeLists.txt` (e.g. `target_include_directories(..., include/prrot)`).  
   - If `engine/src/dsp` was copied, add it to KellyCore or a dedicated target; ensure `rt_harness` and KellyFFI still build.  
   - Add any PRROT lib under existing options (BUILD_RT_HARNESS, BUILD_KELLY_CORE).

2. **Python**  
   - Ensure `music_brain/prrot`, `music_brain/learning`, `music_brain/voice` are on the package path (`pyproject.toml` / `music_brain/__init__.py`).  
   - Keep single entrypoint: `pip install -e .` and `music_brain.api`.

3. **Intent IR**  
   - Align `engine/intent_ir` (if copied) with `engine/intent_ir/src/intent_ir/` and `shared_schemas/`; one schema contract.

4. **Config / scripts**  
   - Review copied configs in `config/` and scripts in `scripts/`; retire or rename duplicates.

5. **AGENTS.md and BUILD.md**  
   - Update repository layout and build options if new dirs (e.g. `include/prrot`, `music_brain/prrot`) or CMake targets appear.  
   - Keep “one UI path” and headless/rt_harness as canonical.

---

## 5. Nested `KmiDi/` (e.g. KmiDi/KmiDi/external/JUCE)

If the repo contains a nested `KmiDi/` (e.g. legacy clone with its own JUCE), treat it as legacy:  
- Prefer a single `external/JUCE` at repo root (per AGENTS).  
- After merging from KmiDi_FINAL, remove or archive the nested `KmiDi/` tree so there is one source of truth for engine, FFI, and UI.

---

## 6. Summary

- **Include:** PRROT, engine/DSP, Intent IR alignment, music_brain learning/voice, configs, scripts.  
- **Exclude:** All UI implementations (standalone app, React from KmiDi_FINAL, plugin editors, Qt/JUCE UI).  
- **Method:** Run `scripts/merge-from-kmidi-final.sh` (Option A in-repo); then resolve build/imports and update AGENTS.md/BUILD.md. Optional: subtree or external worktree if KmiDi_FINAL is a separate repo.

---

## 7. Post-merge audit (main-repo behaviour)

The merge script **overwrites** (does not delete) these destinations when they exist: `include/penta`, `src_penta-core/*`, `engine/src/dsp`, `engine/intent_ir`, `music_brain/prrot`, `music_brain/learning`, `music_brain/voice`, matching `config/*.yaml`, and scripts matching `*robustness*`, `*verify*`, `*build_intent*`.

**Git-history audit (vs current):**

- **music_brain/voice:** Main had `macos_speech`, `neural_voice`, `presets`, and a minimal `__init__` (auto_tune, modulator, synthesizer, voice_classifier). Current tree keeps all main-only files; post-merge `__init__.py` was updated to **re-export** main-only surface so nothing is lost: `MODULATION_PRESETS`, `AUTO_TUNE_PRESETS`, `VOICE_PROFILES`, and optional `MacOSVoice`/`MacOSSpeechSynthesizer`, `UnifiedNeuralVoice`/`NeuralVoiceConfig`/`NeuralVoiceBackend` are now on `music_brain.voice` when available.
- **music_brain/learning:** Current dir is a superset of both main and FINAL (c72f0322 recovery + merge); no behaviour removed.
- **scripts/verify_models.py:** Uses `music_brain.penta_core.ml.model_registry`; no prior main-only version found in history; current script retained.
- **engine/src/dsp, engine/intent_ir, include/penta:** No file deletions; any overwrite replaced content with FINAL’s version. Core KmiDi (React, Tauri, `music_brain.api`) is never touched by the script.

**Restored:** `music_brain/voice/__init__.py` now exposes presets and optional macos_speech/neural_voice so main-repo callers can use `from music_brain.voice import VOICE_PROFILES` (or MacOSVoice, UnifiedNeuralVoice) as before.
