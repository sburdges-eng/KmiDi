# Architecture Follow-Through Execution Plan

> For Hermes: Use subagent-driven-development skill to execute this plan task-by-task. Keep one module/risk surface per task. Respect docs/AGENT_ALLOWED_SURFACES.md and docs/HUMAN_OWNED_SURFACES.md before editing protected surfaces.

Goal: convert the approved architecture handoff into an enforceable implementation-alignment program, with concrete drift findings, file-scoped follow-ups, and test/CI enforcement targets.

Architecture: The architecture interview set is already complete and authoritative in repo docs. This plan does not re-open architecture discovery. It audits the current implementation against the authority set, records drift in a single tracker, and stages follow-up work by risk surface: module map, persistence, Intent IR, native runtime ownership, JUCE/RT, and FFI/ABI.

Tech stack: Markdown docs, React/Vite TypeScript, Python/FastAPI, Rust Intent IR staticlib, C++/JUCE/CMake, pytest, cargo test, CMake/ctest.

---

## Ground rules

- Treat these documents as architectural authority:
  - `docs/ARCHITECTURE.md`
  - `docs/REPO_MODULE_MAP.md`
  - `docs/PERSISTENCE_AND_MIGRATION.md`
  - `docs/INTENT_IR_AUTHORITY.md`
  - `docs/NATIVE_RUNTIME_OWNERSHIP.md`
  - `docs/JUCE_RT_RULES.md`
  - `docs/FFI_OWNERSHIP_AND_ABI.md`
  - `docs/AGENT_ALLOWED_SURFACES.md`
  - `docs/HUMAN_OWNED_SURFACES.md`
- Do not re-derive passes A-G.
- Any proposed change to exported ABI, intent semantics, or persistence model requires explicit human review.
- Native/RT/FFI edits must stay module-scoped and checklist-driven.
- Prefer audit artifacts and issue lists before code changes.

## Primary artifact to create

Create one tracker document first:
- `docs/ARCHITECTURE_ALIGNMENT_TRACKER.md`

This tracker should contain one row per authority surface with these columns:
- authority area
- authority doc
- implementation surface(s)
- current status (`aligned`, `drift`, `unknown`, `historical-only`)
- risk (`P0`, `P1`, `P2`)
- required owner (`agent-safe`, `strict-checklist`, `human-review`)
- evidence
- next action

Suggested sections in the tracker:
1. authority routing / top-level docs
2. Pass B module-map alignment
3. Pass G persistence alignment
4. Intent IR enforcement
5. native runtime ownership/lifetimes
6. JUCE / RT safety
7. FFI / ABI ownership
8. tests / CI enforcement
9. historical doc residuals

---

### Task 1: Create the alignment tracker scaffold

Objective: establish a single durable artifact that future work updates instead of re-scanning from scratch.

Files:
- Create: `docs/ARCHITECTURE_ALIGNMENT_TRACKER.md`
- Read: `docs/ARCHITECTURE.md`
- Read: `docs/REPO_MODULE_MAP.md`
- Read: `docs/PERSISTENCE_AND_MIGRATION.md`
- Read: `docs/INTENT_IR_AUTHORITY.md`
- Read: `docs/NATIVE_RUNTIME_OWNERSHIP.md`
- Read: `docs/JUCE_RT_RULES.md`
- Read: `docs/FFI_OWNERSHIP_AND_ABI.md`
- Read: `docs/AGENT_ALLOWED_SURFACES.md`
- Read: `docs/HUMAN_OWNED_SURFACES.md`

Step 1: Read the authority docs and extract section headings.

Commands:
- `python3 - <<'PY'
from pathlib import Path
for p in [
    'docs/ARCHITECTURE.md',
    'docs/REPO_MODULE_MAP.md',
    'docs/PERSISTENCE_AND_MIGRATION.md',
    'docs/INTENT_IR_AUTHORITY.md',
    'docs/NATIVE_RUNTIME_OWNERSHIP.md',
    'docs/JUCE_RT_RULES.md',
    'docs/FFI_OWNERSHIP_AND_ABI.md',
    'docs/AGENT_ALLOWED_SURFACES.md',
    'docs/HUMAN_OWNED_SURFACES.md',
]:
    print(f'=== {p} ===')
    for line in Path(p).read_text().splitlines():
        if line.startswith('#') or line.startswith('Status:'):
            print(line)
    print()
PY`

Expected: headings/status lines print for each authority doc.

Step 2: Create the tracker skeleton.

Suggested skeleton:

```md
# Architecture Alignment Tracker

Status: working implementation-alignment tracker against the approved authority set
Last updated: 2026-06-09

## Scope
- This tracker records implementation alignment against the approved architecture authority docs.
- It does not re-open or reinterpret workbook passes A-G.

## Authority set
- docs/ARCHITECTURE.md
- docs/REPO_MODULE_MAP.md
- docs/PERSISTENCE_AND_MIGRATION.md
- docs/INTENT_IR_AUTHORITY.md
- docs/NATIVE_RUNTIME_OWNERSHIP.md
- docs/JUCE_RT_RULES.md
- docs/FFI_OWNERSHIP_AND_ABI.md
- docs/AGENT_ALLOWED_SURFACES.md
- docs/HUMAN_OWNED_SURFACES.md

## Alignment matrix
| Area | Authority doc | Implementation surfaces | Status | Risk | Required owner | Evidence | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| authority routing | docs/ARCHITECTURE.md | README.md; QUICK_START.md; BUILD.md; docs/DEVELOPMENT.md; docs/BOOT.md; docs/ENVIRONMENT.md | unknown | P0 | agent-safe | pending | audit and normalize routing notes |
```

Step 3: Save the file.

Run: `git status --short docs/ARCHITECTURE_ALIGNMENT_TRACKER.md`
Expected: file appears as untracked (`??`) or modified (`M`) if re-running.

Step 4: Commit scaffold.

```bash
git add docs/ARCHITECTURE_ALIGNMENT_TRACKER.md
git commit -m "docs: add architecture alignment tracker scaffold"
```

---

### Task 2: Audit authority-routing and top-level doc entrypoints

Objective: ensure high-traffic docs route readers to the authority set instead of reintroducing ambiguity.

Files:
- Modify: `docs/ARCHITECTURE_ALIGNMENT_TRACKER.md`
- Inspect: `README.md`
- Inspect: `QUICK_START.md`
- Inspect: `BUILD.md`
- Inspect: `docs/DEVELOPMENT.md`
- Inspect: `docs/BOOT.md`
- Inspect: `docs/ENVIRONMENT.md`

Step 1: Search for top-level conflict notes and authority routing language.

Commands:
- `rg -n "ARCHITECTURE.md|REPO_MODULE_MAP.md|PERSISTENCE_AND_MIGRATION.md|historical/legacy drift|when docs disagree|authoritative" README.md QUICK_START.md BUILD.md docs/DEVELOPMENT.md docs/BOOT.md docs/ENVIRONMENT.md`

Expected: hits show whether each high-traffic doc points to the current authority set.

Step 2: Record one row per doc in the tracker.

Classification rules:
- `aligned`: clearly routes to authority docs and does not present stale current architecture
- `drift`: still presents conflicting architecture or runnable guidance as current
- `historical-only`: intentionally preserves old wording with explicit historical banner
- `unknown`: not yet reviewed

Step 3: If a top-level doc is missing routing notes, patch only that doc.

Verification:
- `rg -n "ARCHITECTURE.md|REPO_MODULE_MAP.md|PERSISTENCE_AND_MIGRATION.md" README.md QUICK_START.md BUILD.md docs/DEVELOPMENT.md docs/BOOT.md docs/ENVIRONMENT.md`
Expected: every high-traffic doc set has at least one route into the authority set or conflict-resolution instruction.

Step 4: Commit.

```bash
git add docs/ARCHITECTURE_ALIGNMENT_TRACKER.md README.md QUICK_START.md BUILD.md docs/DEVELOPMENT.md docs/BOOT.md docs/ENVIRONMENT.md
git commit -m "docs: align top-level architecture routing"
```

---

### Task 3: Audit Pass B module-map alignment

Objective: compare real repo structure and dependency directions against `docs/REPO_MODULE_MAP.md`.

Files:
- Modify: `docs/ARCHITECTURE_ALIGNMENT_TRACKER.md`
- Inspect: `docs/REPO_MODULE_MAP.md`
- Inspect: `CMakeLists.txt`
- Inspect: `package.json`
- Inspect: `pyproject.toml`
- Inspect: `music_brain/`
- Inspect: `engine/intent_ir/`
- Inspect: `src/`
- Inspect: `src/ui/`
- Inspect: `src_penta-core/`
- Inspect: `include/`
- Inspect: `libs/daiw/`
- Inspect: `python/mcp/`

Step 1: Capture the actual top-level tree.

Commands:
- `python3 - <<'PY'
from pathlib import Path
root = Path('.')
for p in sorted(root.iterdir()):
    if p.name.startswith('.git'):
        continue
    print(p.name + ('/' if p.is_dir() else ''))
PY`

Expected: current top-level tree prints cleanly.

Step 2: Capture declared build/package entry surfaces.

Commands:
- `python3 - <<'PY'
import json
from pathlib import Path
print('package.json scripts:')
pkg = json.loads(Path('package.json').read_text())
for k,v in pkg.get('scripts', {}).items():
    print(f'  {k}: {v}')
print('\npyproject headline:')
for line in Path('pyproject.toml').read_text().splitlines()[:120]:
    print(line)
PY`

Expected: package scripts and pyproject header sections print.

Step 3: Compare actual modules to the Pass B classes in `docs/REPO_MODULE_MAP.md`.

Check explicitly:
- plugin/runtime shell surfaces
- native runtime/engine surfaces
- Intent IR contract surfaces
- backend orchestration surfaces
- generated/schema surfaces
- support/native utility surfaces
- historical/migration/artifact surfaces

Step 4: Record tracker rows and create a short sub-list under Pass B:
- naming/layout drift only
- dependency-direction drift
- ownership-boundary drift
- historical subtree cleanup candidates

Step 5: Optional supporting evidence commands.

Commands:
- `rg -n "shared_schemas|IntentFrame|validate_intent_frame_ffi|KellyFFI|processBlock|PluginProcessor|PluginEditor" src engine include music_brain python/mcp shared_schemas`
- `rg -n "src/ui/|libs/daiw|src_penta-core|python/mcp" docs/REPO_MODULE_MAP.md docs/ARCHITECTURE_DRIFT_AUDIT_2026-06-08.md`

Expected: enough evidence to tie rows to files.

Step 6: Commit.

```bash
git add docs/ARCHITECTURE_ALIGNMENT_TRACKER.md
git commit -m "docs: record pass-b module alignment findings"
```

---

### Task 4: Audit Pass G persistence and migration alignment

Objective: verify save/load authority boundaries and compatibility seams against `docs/PERSISTENCE_AND_MIGRATION.md`.

Files:
- Modify: `docs/ARCHITECTURE_ALIGNMENT_TRACKER.md`
- Inspect: `docs/PERSISTENCE_AND_MIGRATION.md`
- Inspect: plugin/runtime serialization code
- Inspect: project/session state code
- Inspect: any migration/version markers in persistence-related paths
- Inspect: relevant tests under `tests/`

Step 1: Locate persistence and load/save surfaces.

Commands:
- `rg -n "serialize|deserialize|save|load|restore|autosave|migration|version" src engine include tests music_brain --glob '!**/node_modules/**'`

Expected: candidate persistence code paths print.

Step 2: For each persistence path, answer in the tracker:
- what is the canonical persisted truth?
- where is Intent IR persisted?
- what runtime state is reconstructed instead of serialized?
- where does migration/versioning happen?
- are old-save compatibility promises implemented, partial, or absent?

Step 3: If save/load surfaces are spread across multiple layers, create a short “authority mismatch” note with exact files.

Step 4: Verification commands.

Commands:
- `rg -n "schema_version|version|migrate|upgrade" src engine include tests music_brain`
- `python3 -m pytest tests/ -q`

Expected:
- grep identifies version/migration touchpoints
- pytest completes; note failures separately if pre-existing

Step 5: Commit.

```bash
git add docs/ARCHITECTURE_ALIGNMENT_TRACKER.md
git commit -m "docs: record persistence and migration alignment findings"
```

---

### Task 5: Audit Intent IR enforcement paths

Objective: prove where canonical validation happens and identify any bypasses.

Files:
- Modify: `docs/ARCHITECTURE_ALIGNMENT_TRACKER.md`
- Inspect: `shared_schemas/CompleteSongIntentRequest.json`
- Inspect: `scripts/sync_entities.py`
- Inspect: `engine/intent_ir/src/ffi.rs`
- Inspect: `engine/intent_ir/src/`
- Inspect: `src/types/Intent.ts`
- Inspect: `music_brain/api.py`
- Inspect: `music_brain/engine_api/`
- Inspect: `tests/`

Step 1: Trace schema/codegen/validation surfaces.

Commands:
- `rg -n "CompleteSongIntentRequest|IntentFrame|validate_intent_frame_ffi|sync_entities|generated/intent.rs|Intent.ts" shared_schemas scripts engine src music_brain tests`

Expected: full canonical contract path is visible.

Step 2: Record legal flows in tracker.

Required legal flows:
- UI draft -> normalization -> validated Intent IR
- backend request adapter -> validated Intent IR
- persisted intent load -> migration/normalization -> validated Intent IR
- engine-facing requests -> validated Intent IR only

Step 3: Record any bypasses.

A bypass is any path that sends semantic intent meaning into engine-facing or persisted truth without the canonical validation chain.

Step 4: Verification commands.

Commands:
- `python3 scripts/sync_entities.py`
- `cd engine/intent_ir && cargo test`
- `python3 -m pytest tests/ -q`

Expected:
- sync succeeds or reports exact drift
- cargo test passes in `engine/intent_ir`
- pytest completes; note pre-existing failures separately

Step 5: Commit.

```bash
git add docs/ARCHITECTURE_ALIGNMENT_TRACKER.md src/types/Intent.ts engine/intent_ir/src/generated/intent.rs
git commit -m "docs: record intent-ir enforcement findings"
```

Note: only include generated files in the commit if the sync changed them and that change is intentional.

---

### Task 6: Audit native runtime ownership and JUCE/RT lifecycle hazards

Objective: compare actual lifetime and thread-boundary code against `docs/NATIVE_RUNTIME_OWNERSHIP.md` and `docs/JUCE_RT_RULES.md`.

Files:
- Modify: `docs/ARCHITECTURE_ALIGNMENT_TRACKER.md`
- Inspect: `docs/NATIVE_RUNTIME_OWNERSHIP.md`
- Inspect: `docs/JUCE_RT_RULES.md`
- Inspect: plugin processor/editor code
- Inspect: engine runtime/session roots
- Inspect: transport/timeline mutation surfaces
- Inspect: RT queue/snapshot handoff code

Step 1: Locate key ownership classes and RT callback surfaces.

Commands:
- `rg -n "PluginProcessor|PluginEditor|EngineRuntime|EngineSession|processBlock|SafePointer|timerCallback|AsyncUpdater|Transport|Timeline" src engine include src_penta-core libs/daiw`

Expected: file paths for lifecycle and RT-critical classes print.

Step 2: Fill tracker rows for:
- runtime root ownership
- editor/processor relationship
- shutdown ordering hazards
- async callback invalidation
- RT handoff primitives
- forbidden RT behaviors observed or not yet disproven

Step 3: If code changes are proposed later, split them into separate follow-up tasks by one module each.

Step 4: Evidence commands.

Commands:
- `rg -n "new |delete |std::shared_ptr|std::unique_ptr|OwnedArray|ScopedPointer|SafePointer" src engine include src_penta-core libs/daiw`
- `rg -n "mutex|lock_guard|unique_lock|std::lock|malloc|free|new |delete |throw " src engine include src_penta-core libs/daiw`

Expected: search results show ownership patterns and possible RT hazards for manual review.

Step 5: Commit.

```bash
git add docs/ARCHITECTURE_ALIGNMENT_TRACKER.md
git commit -m "docs: record native ownership and rt findings"
```

---

### Task 7: Audit FFI / ABI / cross-language ownership contracts

Objective: verify symbol-level ownership/error/lifetime posture against `docs/FFI_OWNERSHIP_AND_ABI.md`.

Files:
- Modify: `docs/ARCHITECTURE_ALIGNMENT_TRACKER.md`
- Inspect: `docs/FFI_OWNERSHIP_AND_ABI.md`
- Inspect: `src/bridge/kelly_ffi.h`
- Inspect: C++ FFI bridge implementation files
- Inspect: `engine/intent_ir/src/ffi.rs`
- Inspect: Python binding/consumer surfaces
- Inspect: native tests/benchmarks

Step 1: Locate exported symbols and free/destroy pairs.

Commands:
- `rg -n "^.*kelly_.*\(|IntentFrameBuilder_|validate_intent_frame_ffi|destroy|free" src/bridge engine/intent_ir include tests`

Expected: exported symbols and destroy/free surfaces print.

Step 2: Build a tracker subtable for each exported family:
- symbol family
- return style (`opaque handle`, `caller-owned buffer`, `callee-allocated snapshot`, `borrowed`, `static data`)
- matching free/destroy pair
- documented invalidation rule
- thread-affinity note
- drift/risk status

Step 3: Verify panic/exception containment.

Commands:
- `rg -n "panic|catch_unwind|throw|noexcept" engine/intent_ir src/bridge include`

Expected: enough evidence to evaluate boundary safety posture.

Step 4: Optional native verification commands.

Commands:
- `cd engine/intent_ir && cargo test`
- `cmake -S . -B build -G Ninja -DBUILD_KELLY_FFI=ON -DBUILD_KELLY_CORE=ON -DBUILD_TESTS=ON`
- `cmake --build build --target KellyFFI -j8`

Expected:
- Rust tests pass
- CMake configure succeeds if toolchain/deps are present
- KellyFFI builds or fails with actionable dependency info

Step 5: Commit.

```bash
git add docs/ARCHITECTURE_ALIGNMENT_TRACKER.md
git commit -m "docs: record ffi ownership and abi findings"
```

---

### Task 8: Build the rule-to-enforcement matrix

Objective: ensure authority rules have tests, checks, or explicit missing-coverage entries.

Files:
- Modify: `docs/ARCHITECTURE_ALIGNMENT_TRACKER.md`
- Inspect: `tests/`
- Inspect: `pytest.ini`
- Inspect: CMake test config
- Inspect: schema-sync and native verification docs/scripts

Step 1: Create a matrix section in the tracker with columns:
- rule
- authority doc
- current enforcement command/test
- status (`covered`, `partial`, `missing`)
- next action

Seed rows:
- schema drift check
- Rust validator tests
- Python schema tests
- persistence compatibility tests
- native sanitizer run
- RT regression harness
- FFI ownership regression tests
- plugin load/open/close smoke tests

Step 2: Verification commands.

Commands:
- `python3 -m pytest tests/ -q`
- `cd engine/intent_ir && cargo test`
- `ctest --test-dir build --output-on-failure`

Expected:
- each command either passes or gives exact missing-coverage / environment information to record

Step 3: Commit.

```bash
git add docs/ARCHITECTURE_ALIGNMENT_TRACKER.md
git commit -m "docs: add architecture enforcement coverage matrix"
```

---

### Task 9: Produce the follow-up queue from tracker findings

Objective: convert the tracker into bounded execution slices without mixing risk surfaces.

Files:
- Modify: `docs/ARCHITECTURE_ALIGNMENT_TRACKER.md`
- Optionally create: `docs/handoffs/architecture-alignment-followups.md` if the repo adopts a handoff directory later

Step 1: Add a final section:
- P0 strict-checklist fixes
- P0 human-review-required items
- P1 agent-safe cleanup
- historical-only residuals

Step 2: Each follow-up item must include:
- title
- exact file/module scope
- owner type
- required verification command
- blocked-by note if human review is needed first

Step 3: Commit.

```bash
git add docs/ARCHITECTURE_ALIGNMENT_TRACKER.md docs/handoffs/architecture-alignment-followups.md 2>/dev/null || git add docs/ARCHITECTURE_ALIGNMENT_TRACKER.md
git commit -m "docs: add architecture alignment follow-up queue"
```

---

## Verification checklist for the whole plan

Run these as applicable during execution:

```bash
python3 -m pytest tests/ -q
cd engine/intent_ir && cargo test
cmake --build build --target KellyFFI -j8
ctest --test-dir build --output-on-failure
```

Expected outcomes:
- schema and Python checks identify contract drift if present
- Rust tests validate Intent IR boundary rules
- native build/test commands validate whether the current tree still matches the documented native/FFI architecture

## Success criteria

This plan is successful when:
- `docs/ARCHITECTURE_ALIGNMENT_TRACKER.md` exists and is populated with evidence-backed rows
- every major authority doc has at least one corresponding tracker section
- Pass B and Pass G implementation drift are concretely recorded
- Intent IR, native runtime ownership, JUCE/RT, and FFI/ABI risks are captured with exact file evidence
- follow-up tasks are split by bounded module/risk surface
- human-review-required items are clearly separated from agent-safe cleanup

## Notes for execution

- Start with docs and evidence collection; do not jump straight into protected-surface code edits.
- If a search returns zero results where a surface should exist, broaden the search before concluding the surface is absent.
- If build/test commands fail because of environment/toolchain gaps, record that as evidence instead of silently skipping.
- Do not mix persistence-model changes, ABI changes, and RT ownership refactors into one task.
