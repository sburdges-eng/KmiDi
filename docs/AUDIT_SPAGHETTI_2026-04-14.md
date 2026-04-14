# KmiDi Spaghetti Code Audit — 2026-04-14

> **Scope:** `music_brain/` and `bindings/` primary; `src/`, `include/`, `src_penta-core/` secondary.
> **Excluded:** `_archive/`, `external/`, `_deps/`, `.worktrees/`, `*_archive*`, test files.
> **Mode:** READ-ONLY — no code was modified.

---

## Phase 1 — Evidence (Ranked Hotlist)

### TOP-10 FINDINGS

#### 1. `api.py` + `api_misc.py`: Full-file Clone (blast radius: **cross-module**)

| Path | LOC |
|------|-----|
| `music_brain/api.py` | 1912 |
| `music_brain/api_misc.py` | 576 |
| `music_brain/misc_code/api.py` | ~576 (import-broken copy) |

**Diagnosis:** `api_misc.py` is a character-for-character earlier revision of the `DAiWAPI` class now in `api.py`. Every method — `generate_harmony_from_intent`, `generate_basic_progression`, `humanize_drums`, `therapy_session`, etc. — exists in both files with the same signatures and (in `api_misc.py`) no safe-access guards.

- `api.py:75–805` vs `api_misc.py:59–520` — identical method bodies for 12+ methods.
- `api_misc.py:17` imports `from music_brain.harmony import (HarmonyGenerator, generate_midi_from_harmony)`, referencing a shim that re-exports from `kelly_companion`. `api.py:38` imports from `music_brain.harmony_utils`.
- `misc_code/api.py` is yet another copy with a broken import path (`from music_brain.harmony import ...`).
- `api_misc.py:572` and `api.py:866` both export a global `api = DAiWAPI()` singleton.

**Dead-code verdict:** `api_misc.py` is not imported by any live module. `misc_code/api.py` is not imported by any live module. Both are dead.

---

#### 2. 12× `except Exception: logging.exception(…); raise HTTPException(500)` in `api.py` (blast radius: **module**)

Locations (all in `music_brain/api.py`):

| Line | Endpoint |
|------|----------|
| 1045–1047 | `/emotions` |
| 1185–1187 | `reload_humanizer` |
| 1214–1216 | `spectocloud import` |
| 1317–1319 | `spectocloud render` |
| 1636–1638 | `_handle_generate_music` |
| 1688–1690 | `/interrogate` |
| 1701–1703 | `POST /lyrics` |
| 1713–1715 | `GET /lyrics` |
| 1763–1765 | `/audio/classify` |
| 1804–1806 | `/audio/valence-arousal` |
| 1830–1832 | `/audio/models` |
| 1847–1849 | `/voice/classify` |

Pattern:
```python
except Exception:   # pragma: no cover
    logging.exception("<endpoint> failed")
    raise HTTPException(status_code=500, detail=_HTTP_500_DETAIL)
```

Some use the `_HTTP_500_DETAIL` constant, some hardcode `"Internal server error."` — an ad-hoc inconsistency.

---

#### 3. `harmony_utils/harmony_generator.py` ↔ `data_utils/harmony_generator.py`: Near-100% duplicate (blast radius: **cross-module**)

| File | LOC | Δ from canonical |
|------|-----|------------------|
| `music_brain/harmony_utils/harmony_generator.py` | 541 | Canonical (used by `api.py`) |
| `music_brain/data_utils/harmony_generator.py` | 553 | +12 LOC: adds `key.upper()` normalization, `mido` fallback |

Both define: `RuleBreakType`, `ChordVoicing`, `HarmonyResult`, `HarmonyGenerator`, `generate_midi_from_harmony`.
Both define identical `NOTE_TO_MIDI`, `SCALES`, `DIATONIC_CHORDS_MAJOR`, `DIATONIC_CHORDS_MINOR` dicts.

`data_utils` version has two real improvements:
- `_roman_to_chord_symbols` strips trailing "m" from key input: `key.upper()` (line 216–217).
- `generate_midi_from_harmony` has `try/except ImportError` for `mido` (line 460–467) — the canonical one hard-crashes.

**Divergence risk:** Whoever next fixes flat-key enharmonic corruption will fix it in one and forget the other.

---

#### 4. 4-way Harmony Module Sprawl (blast radius: **cross-module**)

| Module | Purpose | Imported by live code? |
|--------|---------|----------------------|
| `music_brain/harmony/` | Shim → `kelly_companion.utils.harmony_system` | `api_misc.py` (dead); NOT by `api.py` |
| `music_brain/harmony.py` | `generate_chords_with_learning()` | No live importer found |
| `music_brain/harmony_utils/` | `HarmonyGenerator`, `HarmonyResult`, `generate_midi_from_harmony` | **`api.py:38–44` (live)** |
| `music_brain/harmony_kmidi.py` | MCP tools wrapping `data.harmony_generator` | Unclear (MCP registration) |
| `music_brain/orchestrator/processors/harmony.py` | `HarmonyProcessor` for pipeline | `bridge_api.py` (live) |
| `music_brain/data_utils/harmony_generator.py` | Duplicate of `harmony_utils` | `harmony_kmidi.py:10` |

Two classes named `HarmonyGenerator` (in `harmony_utils` and `data_utils`). Two classes named `HarmonyResult`. `harmony_kmidi.py:10` imports from `data.harmony_generator` — a path that only resolves if a `data/` package sits at the Python path root.

---

#### 5. `IntentPipeline` (intent_pipeline.py) vs `Pipeline` (orchestrator/pipeline.py): Two Parallel Pipeline Abstractions (blast radius: **cross-process**)

| Concept | `pipeline/intent_pipeline.py` | `orchestrator/pipeline.py` |
|---------|-------------------------------|---------------------------|
| Pipeline class | `IntentPipeline` (3-stage sync: normalize→validate→expand) | `Pipeline` (N-stage async: ProcessorInterface stages) |
| Intent normalization | `IntentPipeline.normalize()` lines 244–316 | `IntentProcessor._process_impl()` in `processors/intent.py` |
| Harmony dispatch | Delegated to `process_intent()` via `api.py` | `HarmonyProcessor._generate_harmony()` in `processors/harmony.py` |
| Groove | Not in pipeline | `GrooveProcessor` stage |
| Used by | `/generate` HTTP endpoint (`api.py:1339`) | `bridge_api.process_prompt()` (C++ bridge, line 572–574) |
| Validation | Pydantic `CompleteSongIntentRequest` | `ProcessorInterface.validate_input()` |

Both pipelines do emotion-to-key mapping, both clamp BPM to 40–300, both normalize structure lists. The orchestrator Pipeline has retry/timeout/callback infrastructure that the IntentPipeline lacks.

---

#### 6. 8× Pydantic v1/v2 Compat Shim `if hasattr(x, "model_dump") … elif hasattr(x, "dict")` (blast radius: **module**)

All in `music_brain/pipeline/intent_pipeline.py`:
- Lines 73–79: `_tech_dict()`
- Lines 145–148: `_normalize_structure()` (item serialization)
- Lines 169–172: `_normalize_instruments()` (item serialization)
- Lines 194–197: `_orchestration_present()` (raw serialization)
- Lines 207–210: `_structure_from_ttg()` (timeline_raw)
- Lines 218–221: `_instruments_from_orchestration()` (orch)
- Lines 310–313: energy_curve serialization

Also in `api.py`:
- Lines 1422–1432: Structure/instruments serialization (2× occurrences)
- Lines 1493–1504: Duplicate of the above in response building

---

#### 7. `VALID_MUSICAL_MODES` Defined in Two Places (blast radius: **isolated**)

- `music_brain/api.py:29–30`: `VALID_MUSICAL_MODES = {"major", "minor", "dorian", …}`
- `music_brain/pipeline/intent_pipeline.py:23–26`: Identical set.

Only `intent_pipeline.py` actually uses it (line 120). `api.py` defines it but never references it in the 1912-line file.

---

#### 8. `NOTE_TO_MIDI` / `root_to_midi` Defined in 3+ Locations (blast radius: **cross-module**)

| File:line | Name | Notes |
|-----------|------|-------|
| `harmony_utils/harmony_generator.py:57` | `NOTE_TO_MIDI` | Class-level dict |
| `data_utils/harmony_generator.py:58` | `NOTE_TO_MIDI` | Identical copy |
| `orchestrator/bridge_api.py:420` | `root_to_midi` | Local dict, same map but with MIDI values offset by +60 (middle C) |
| `data_utils/chord_diagnostics.py:64` | `NOTE_TO_MIDI` | Another copy |

These must stay synchronized. Any enharmonic fix (e.g. `F#` vs `Gb`) must be applied to all.

---

#### 9. `api.py` `_handle_generate_music()` — 238-line Function (Lines 1321–1559) (blast radius: **module**)

- LOC: 238 (exceeds 80 LOC threshold by 3×)
- Nesting depth: 5 (try → if → if → try → if)
- Mixes: validation, BPM clamping, structure calculation, HarmonyPlan construction, MIDI rendering, audio rendering, response assembly, 2× duplicate `model_dump`/`dict` serialization of structure/instruments
- Contains inline `import tempfile; import time` (lines 1372–1373) despite both being imported at file top

---

#### 10. `harmony_utils/harmony_generator.py:485–489` — `print()` in Library Code (blast radius: **isolated**)

```python
print(f"MIDI file saved: {output_path}")
print(f"Key: {harmony.key} {harmony.mode}")
print(f"Chords: {' - '.join(harmony.chords)}")
```

Also in `data_utils/harmony_generator.py:497–502` (the duplicate). Library code should use `logging`, not `print()`.

---

### APPENDIX — Additional Findings

#### A. Length & Complexity

| File | LOC | Concern |
|------|-----|---------|
| `api.py` | 1912 | God-file: API class (866 LOC) + HTTP routes (1046 LOC) in one file |
| `orchestrator/bridge_api.py` | 676 | Mixes: synesthesia dict, genre detection, MIDI generation, parameter safety |
| `session/teaching.py` | 30+ `print()` calls | Console UI in library code |
| `emotion/multimodal_emotion.py` | 18 `print()` calls | Demo code in module |

#### B. Hidden Coupling & State

| Finding | Path:line |
|---------|-----------|
| Module-level singletons | `api.py:866` — `api = DAiWAPI()` (mutable); `api_misc.py:573` — same |
| Module-level mutable cache | `bridge_api.py:222` — `_genre_definitions: Dict = {}` (global) |
| Functions that compute AND perform I/O | `harmony_generator.py:446` — `generate_midi_from_harmony()` computes MIDI AND writes file AND prints |
| Silent exception in build_humanizer | `api.py:184` — `except Exception: logging.exception(…)` then silently falls back |

#### C. Near-Duplicates (Beyond Top-10)

| Pattern | Instances | Locations |
|---------|-----------|-----------|
| `api_misc.py` DAiWAPI class | 2 | `api.py:159–862`, `api_misc.py:59–573` |
| Chord interval tables (Major/Minor triads) | 3 | `harmony_utils/harmony_generator.py:76–94`, `data_utils/harmony_generator.py:77–95`, `bridge_api.py:444–455` |
| Emotion-to-parameter mapping | 2 | `bridge_api.py:87–121` (`_synesthesia_dictionary`), `data_utils/emotional_mapping.py` (`EMOTIONAL_PRESETS`) |
| Output format → temp file path generation | 2 | `api.py:1374–1387` and `api.py:1577–1594` (full pipeline path and legacy fallback path) |

#### D. Debuggability Hazards

| Hazard | Location |
|--------|----------|
| `print()` instead of `logging` | `harmony_utils/harmony_generator.py:485–489`, `data_utils/harmony_generator.py:497–502`, `session/teaching.py:257–353` (30 calls), `emotion/multimodal_emotion.py:634–681` (18 calls) |
| Catch-all + no re-raise | `bridge_api.py:652` — `except Exception as e: return BridgeResult(success=False, error_message=str(e))` swallows full stack |
| `except (KeyError, ValueError): modified_progression = base_progression` | `harmony_generator.py:138-141` — silent fallback on invalid rule-break name, no log |
| Untyped dict "structs" | `api.py:1479–1533` — response dict built ad-hoc with string keys; no schema validates the output shape |
| Magic string duplication | `"Internal server error."` literal at `api.py:1765,1806,1849` vs `_HTTP_500_DETAIL` at `api.py:890,1047,1187,1216,1319,1638,1690,1703,1715,1832` |

#### E. Dead Code

| Module | Bytes | Evidence of non-use |
|--------|-------|-------------------|
| `music_brain/api_misc.py` | 17,320 | Not imported by any live module |
| `music_brain/misc_code/` (31 files) | ~350 KB total | Contains `api.py`, `analyzer.py`, `effects.py`, `parrot.py` — none imported by live stack |
| `music_brain/harmony.py` | 1,065 | No live importer; only file with `generate_chords_with_learning()` |
| `music_brain/harmony/` | ~600 | Shim re-exporting `IntelligentHarmonySystem` — only imported by dead `api_misc.py` |
| `music_brain/harmony_kmidi.py` | 5,074 | Imports from `data.harmony_generator` (broken path); unclear if MCP server is live |

---

## Phase 2 — Generalization Candidates

### Proposal 1: `pydantic_to_dict(obj) → dict`

| Attribute | Value |
|-----------|-------|
| **Name** | `pydantic_to_dict` (one-liner helper) |
| **Replaces** | 8+ call sites: `intent_pipeline.py:73–79`, `145–148`, `169–172`, `194–197`, `207–210`, `218–221`, `310–313`; `api.py:1422–1432`, `1493–1504` |
| **Signature** | `(obj: Any) → dict` — calls `model_dump()` / `dict()` / `__dict__` |
| **Existing reuse** | `_tech_dict()` in `intent_pipeline.py:70–79` already implements this for one arg. Generalize it. |
| **Debuggability** | Eliminates 8 divergent serialization paths; one place to add logging or breakpoint |
| **Risk** | Zero behavior change — pure extract-method |
| **Reuses or obsoletes** | Adapts `_tech_dict()`, replaces all inline `hasattr(x, "model_dump")` checks |

---

### Proposal 2: `api_error_handler` Decorator

| Attribute | Value |
|-----------|-------|
| **Name** | `@api_error_handler(status_code=500, detail=_HTTP_500_DETAIL)` |
| **Replaces** | 12 `except Exception: logging.exception(…); raise HTTPException(500)` blocks in `api.py` (lines 1045, 1185, 1216, 1319, 1638, 1688, 1701, 1713, 1765, 1806, 1832, 1849) |
| **Signature** | Decorator: `(f: Callable) → Callable` wrapping async endpoints |
| **Existing reuse** | None — but FastAPI has middleware / exception_handler patterns. A decorator is simpler and more explicit. |
| **Debuggability** | All 500-error logging flows through one path. Can add request-id, timing, structured fields once. |
| **Risk** | Low — wraps existing endpoints without changing behavior. Must preserve `HTTPException` pass-through. |
| **Reuses or obsoletes** | N/A — new, but replaces boilerplate, not logic |

---

### Proposal 3: Canonical `NOTE_TO_MIDI` Module

| Attribute | Value |
|-----------|-------|
| **Name** | `music_brain.theory.constants` (or add to existing `music_brain/theory/`) |
| **Replaces** | 4 definitions: `harmony_utils/harmony_generator.py:57`, `data_utils/harmony_generator.py:58`, `bridge_api.py:420`, `data_utils/chord_diagnostics.py:64` |
| **Signature** | `NOTE_TO_MIDI: Dict[str, int]` (pitch-class 0–11); `note_to_absolute_midi(name: str, octave: int = 4) → int` |
| **Existing reuse** | `music_brain/theory/` exists as a package — check if constants are already there |
| **Debuggability** | Enharmonic fixes (F# vs Gb) applied once, everywhere |
| **Risk** | Low — pure constant extraction. Each consumer replaces dict literal with import. |
| **Reuses or obsoletes** | Obsoletes all 4 inline dicts |

---

### Proposal 4: Consolidate `harmony_utils/harmony_generator.py` ← `data_utils/harmony_generator.py`

| Attribute | Value |
|-----------|-------|
| **Name** | Keep `harmony_utils/harmony_generator.py` as canonical |
| **Replaces** | `data_utils/harmony_generator.py` (553 LOC, 99% identical) |
| **Signature** | No change — merge the 2 improvements from `data_utils` version into canonical |
| **Existing reuse** | `api.py` already imports from `harmony_utils`. |
| **Debuggability** | Eliminates shadow divergence; `harmony_kmidi.py` import `from data.harmony_generator` must be fixed to `from music_brain.harmony_utils.harmony_generator` |
| **Risk** | `harmony_kmidi.py` import path changes; test with `python -c "from music_brain.harmony_kmidi import register_tools"`. `data_utils/harmony_generator.py` can leave a deprecation shim. |
| **Reuses or obsoletes** | Obsoletes `data_utils/harmony_generator.py` |

---

### Proposal 5: Delete Dead Code Modules

| Attribute | Value |
|-----------|-------|
| **Name** | Dead code deletion |
| **Replaces** | `api_misc.py`, `misc_code/` (31 files), `music_brain/harmony.py`, `music_brain/harmony/` (shim) |
| **Signature** | N/A |
| **Existing reuse** | The live stack imports from `harmony_utils`, not `harmony` or `api_misc` |
| **Debuggability** | Grep results no longer polluted with dead files; IDE completion stops suggesting stale symbols |
| **Risk** | Medium — must verify no external tool or script imports these. `grep -r "from music_brain.api_misc\|from music_brain.misc_code\|from music_brain.harmony import\|from music_brain.legacy"` in the full repo (excluding `_archive/`). Current scan shows zero live importers. |
| **Reuses or obsoletes** | Obsoletes `api_misc.py`, `misc_code/`, `harmony.py`, `harmony/` |

---

### Proposal 6: Extract `_handle_generate_music()` into Sub-Functions

| Attribute | Value |
|-----------|-------|
| **Name** | Split into: `_build_harmony_plan()`, `_render_outputs()`, `_assemble_response()` |
| **Replaces** | `api.py:1321–1559` (238-LOC monster) |
| **Signature** | Each sub takes typed args, returns typed result. No behavioral change. |
| **Existing reuse** | `HarmonyPlan` already exists; the construction code at lines 1446–1458 should be a factory. |
| **Debuggability** | Stack traces point to specific sub-function instead of a 238-line block. Each sub can be unit-tested independently. |
| **Risk** | Low — pure extract-method refactor. Test with existing `/generate` integration test. |
| **Reuses or obsoletes** | N/A |

---

### Proposal 7: Unify `VALID_MUSICAL_MODES` Constant

| Attribute | Value |
|-----------|-------|
| **Name** | Move to `music_brain.theory.constants` (same module as Proposal 3) |
| **Replaces** | `api.py:29` (unused there), `intent_pipeline.py:23` |
| **Signature** | `VALID_MUSICAL_MODES: FrozenSet[str]` |
| **Existing reuse** | Could live in `music_brain/theory/` |
| **Debuggability** | One source of truth for valid modes |
| **Risk** | Zero — `api.py` doesn't use it; `intent_pipeline.py` imports from new location |
| **Reuses or obsoletes** | N/A |

---

## Phase 3 — Refactor Sequencing

Each step is independently revertable and ordered so no step depends on later work.

### Mechanical Steps (No Behavior Change, No New Tests Needed)

| # | Proposal | Diff Size | Dependencies |
|---|----------|-----------|-------------|
| 1 | **Delete dead code** (Proposal 5): remove `api_misc.py`, `misc_code/`, `music_brain/harmony.py`, `music_brain/harmony/` | ~500 LOC deleted | None — verify zero importers first |
| 2 | **Extract `pydantic_to_dict()`** (Proposal 1): add helper to `music_brain/utils/`, update `intent_pipeline.py` and `api.py` | ~80 LOC | None |
| 3 | **Unify `VALID_MUSICAL_MODES`** (Proposal 7): move to `theory/constants.py`, update `intent_pipeline.py`, remove from `api.py` | ~15 LOC | None |
| 4 | **Extract `NOTE_TO_MIDI`** (Proposal 3): create `theory/constants.py`, update 4 consumers | ~50 LOC | Step 3 (same target file) |
| 5 | **`@api_error_handler` decorator** (Proposal 2): add to `api.py`, apply to 12 endpoints | ~120 LOC | None |
| 6 | **Extract `_handle_generate_music` sub-functions** (Proposal 6): split into 3 helpers | ~200 LOC (moved, net neutral) | Step 2 (uses `pydantic_to_dict`) |

### Steps Requiring Behavior Change or New Tests

| # | Proposal | Diff Size | Risk | Needs |
|---|----------|-----------|------|-------|
| 7 | **Consolidate harmony generators** (Proposal 4): merge `data_utils` improvements into `harmony_utils`, fix `harmony_kmidi.py` import, add deprecation shim | ~60 LOC | Medium — `harmony_kmidi.py` import path changes | Test: `python -c "from music_brain.harmony_kmidi import register_tools"`; verify `mido` fallback works |
| 8 | **Replace `print()` with `logging`** in `harmony_generator.py`, `teaching.py`, `multimodal_emotion.py` | ~50 LOC | Low — output channel changes | Manual verification that no downstream tool parses stdout |

### Deferred (Requires Design Decision)

| Item | Why Deferred |
|------|-------------|
| Merge `IntentPipeline` + `orchestrator/Pipeline` | Architectural: these serve different entry points (HTTP vs C++ bridge). Need to decide on a single pipeline abstraction first. |
| Migrate `bridge_api.py` synesthesia/genre dicts to config files | Needs product decision on whether these are user-editable |
| Refactor `DAiWAPI` class to separate intent-processing from I/O wrappers | Would touch every method; needs API stability contract |

---

> **Next step:** Review this report. After approval, each numbered step in Phase 3 becomes its own focused prompt: *"Apply Phase 3 step #1 — delete dead code modules."*
