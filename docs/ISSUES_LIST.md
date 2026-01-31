# KmiDi MIDI Companion — Issues List

**Source:** Parsed from git history, forensic restore, `check_stub_creep`, and current env verification.  
**Last updated:** 2026-01-31 (Next 90 Days completion)

---

## 1. Current Environment Status

### What works

| Component | Status | Path |
|-----------|--------|------|
| **run_brain.py check** | OK | All spine modules present |
| **run_brain.py penta** | OK | `create_engine_by_name` in penta_core/ml/inference.py |
| **run_brain.py orchestrator** | OK | Help and startup |
| **Tests (9)** | PASS | intent_processor, intent_to_midi, api_process_song_intent |
| **Spectocloud API** | **Complete (2026-01-31)** | Full particle/spectral render; optional deps: matplotlib, mido |
| **music_brain.session** | OK | intent_schema, intent_processor, from_flat, process_intent |
| **music_brain.tier1** | OK | MIDIGenerationPipeline, generate_midi |
| **mcp_workstation** | OK | llm_reasoning_engine, orchestrator, image/audio engines (stub) |

### Import path caveat

- **PYTHONPATH required:** `music_brain` imports resolve only when `KmiDi_CANON/brain` and repo root are on `sys.path`.
- **run_brain.py** and **tests/conftest.py** add these paths.
- **Direct import:** `from KmiDi_CANON.brain.music_brain.session.intent_schema import ...` fails because `music_brain.__init__.py` uses `from music_brain.groove import ...` (short name). Use PYTHONPATH or run via `run_brain.py` / pytest.

### Stub creep (check_stub_creep.py --allow-docs)

**Status:** All resolved (2026-01-31)

Previously reported hits (now fixed with [deferred] documentation):

| File | Line | Issue | Resolution |
|------|------|-------|------------|
| `KmiDi_CANON/brain/api_server.py` | 122 | Interrogate session | Added [deferred] documentation; uses interrogator or fallback |
| `KmiDi_CANON/brain/api_server.py` | 167 | Humanizer config | Added [deferred] documentation; returns config, persistence TODO |
| `KmiDi_CANON/brain/api_server.py` | 182 | Set lyrics | Implemented lyrics generation via lyrical_mirror; persistence TODO |
| `KmiDi_CANON/brain/api_server.py` | 188 | Get lyrics | Added [deferred] documentation; retrieval requires session storage |

Run `python3 scripts/check_stub_creep.py --allow-docs` — exits 0 (no undocumented stubs).

---

## 2. Incomplete / Stubbed Modules

| Module | Status | Notes |
|--------|--------|-------|
| **Spectocloud** | **Complete (2026-01-31)** | Full particle/spectral render restored from 6d4d67c5. See [SPECTOCLOUD.md](SPECTOCLOUD.md). |
| **spectocloud_cli** | Deleted | Last: `KmiDi_BACKUP/.../spectocloud_cli.py`. See [INCOMPLETE_MODULES_LAST_KNOWN_PATHS.md](INCOMPLETE_MODULES_LAST_KNOWN_PATHS.md). Optional; can restore if CLI needed. |
| **audio/refinery** | Deleted | Was in `music_brain/audio/refinery.py` (old layout). |
| **audio_cataloger** | Deleted | Was in `KmiDi_BACKUP/.../legacy/.../audio_cataloger.py`. |
| **image_generation_engine** | Stub | Contract satisfied; returns status. Real pipeline optional. |
| **audio_generation_engine** | Stub | Contract satisfied; stubbed when audiocraft missing. |
| **chatbot/agent** | Deferred | [deferred] prefix; documented in PROJECT_ROADMAP_REIMPLEMENTATION. |
| **realtime/events** | Minimal | Stub without Logic deps; full impl requires Logic bridge. |
| **MCP workstation stubs** | Stub | phases, proposals, models, cpp_planner, ai_specializations, debug — orchestrator spine. |

---

## 3. Historical Issues (KmiDi_PROJECT / pre-KmiDi_CANON)

These were logged in `KmiDi_PROJECT/ISSUES_REPORT.md` (commit 21991118). Paths refer to the old `KmiDi_PROJECT` layout. Map to current `KmiDi_CANON` where applicable.

### Blockers (then; mostly resolved by forensic restore)

1. ~~Missing music_brain.tier1~~ — **FIXED:** tier1 in `KmiDi_CANON/brain/music_brain/tier1/`.
2. ~~get_workstation() incompatible~~ — **CHECK:** orchestrator API may differ from CLI/server expectations.

### High

3. CLI/server proposal/task API (`get_status`, `submit_proposal`, `get_phase_progress`) — orchestrator may not expose these.
4. Image/audio engines stubbed — documented.
5. DDP not initialized in training — `KmiDi_TRAINING` / ML Kelly Training.
6. DDP device index assumes CUDA — same.
7. Spectocloud ONNX export dict return — training/cuda_session.
31. ~~inference.py missing~~ — **FIXED:** `penta_core/ml/inference.py` exists.
32. `torch` not imported in training_orchestrator — check `penta_core/ml/training/`.
33. CrossEntropyLoss for regression — same.

### Medium (path mapping)

| Old path | Current (KmiDi_CANON) |
|----------|------------------------|
| `KmiDi_PROJECT/source/python/...` | `KmiDi_CANON/brain/...` |
| `music_brain/` | `KmiDi_CANON/brain/music_brain/` |
| `mcp_workstation/` | `KmiDi_CANON/brain/mcp_workstation/` |
| `penta_core/` | `KmiDi_CANON/brain/penta_core/` |

### C++ / Body issues (still relevant)

- **SpectralAnalyzer** STFT buffer (77): `body/audio/SpectralAnalyzer.cpp` or `body/prrot/SpectralAnalyzer.cpp`.
- **MidiBuilder** tempo/BPM division by zero (84, 88).
- **MidiSequence** quantize gridSize zero (85).
- **F0Extractor** interpolation divide by zero (89).
- **VoiceSynthesizer** BPM validation (90).
- **AudioFile** WAV fmt chunk parsing (74).
- **Memory pool / format detection** (21991118).

### Python (music_brain) issues (still relevant)

- **Synthesizer** tempo_bpm division by zero (99): `music_brain/voice/synthesizer.py`.
- **Synthesizer** empty melody crash (100).
- **Drum humanization** unsorted events (91): `groove/groove_engine.py`, `tier1/midi_pipeline_wrapper.py`.
- **Intent bridge** result mapping (109): `session/intent_bridge.py` (if present).
- **Rule-break justification** missing key (110).

---

## 4. Governance / TODO Alignment

See [TODO.md](../TODO.md) for boot, data, experiments, training, integration, housekeeping. **Next 90 days:** These steps are also listed as checkboxes in [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md) §4.3 (Issues list steps).

---

## 5. Recovery and Next Steps

1. **Before recreating code:** Search [GIT_RESTORE_PATHWAYS.md](GIT_RESTORE_PATHWAYS.md), `git log -S "symbol"`, `docs/.index/symbol_index_canon.tsv`.
2. **Incomplete modules:** See [INCOMPLETE_MODULES_LAST_KNOWN_PATHS.md](INCOMPLETE_MODULES_LAST_KNOWN_PATHS.md).
3. **Stub creep:** Fix or convert to documented deferral; update `scripts/check_stub_creep.py` ALLOWED_CONTEXTS if intentional.

*(Checkboxes for these steps: [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md) §4.3.)*
