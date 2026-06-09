# Architecture Drift Audit

Status: read-only drift audit against approved workbook passes A, B, C, D, E, F, and G
Date: 2026-06-08

Historical handling note
- This audit is preserved as evidence of repo drift observed on 2026-06-08.
- Quoted stale Tauri-era lines below are audit evidence, not current runnable-command or architecture authority.
- When quoted examples conflict with the current repo state, follow the 2026 authority set and checked-in root scripts.

Purpose
- Compare current repository surfaces against the newly approved architecture handoff.
- Identify the highest-signal places where implementation layout, build mechanics, or older docs may confuse future work.
- Distinguish immediate architecture conflicts from tolerated support/layout debt.

Scope
- Read-only audit from repo docs and build configuration.
- No claim that every implementation file was exhaustively reviewed.
- Focused on module mapping, authority layers, and obvious repo-level drift hazards.

## Bottom line

The new architecture handoff is internally consistent, but the repository still contains historical and build-level drift that can mislead future work unless treated explicitly as non-authoritative.

Most important findings:
1. Historical Tauri-centered docs remain in-tree and still describe non-canonical architecture.
2. Native support surfaces are mechanically important in the build (`src_penta-core/`, `libs/daiw/`) and therefore must be treated carefully, even though Pass B correctly classifies them as support rather than independent authority layers.
3. KellyCore is still assembled from a broad `src/*.cpp` glob plus exclusion filters, which means module boundaries are only partially enforced mechanically.
4. Plugin sources are excluded from KellyCore, but `src/ui/` remains part of the broad core source set, so the approved presentation-under-plugin-runtime boundary is not yet strongly enforced by target structure.

## Findings

### 1. Historical Tauri-centered docs remain in tree

Finding
- Multiple docs still describe Tauri as a current runtime shell or architecture center, which conflicts with the new canonical handoff.

Evidence
- `docs/UI_BOUNDARY_RULES.md:161` -> `**Technology:** React + Tauri`
- `docs/WORKSPACE_SETUP.md:105` -> `engine/intent_ir/             # Tauri backend (Rust)`
- `docs/BOOT.md:19` -> `Tauri/React desktop shell optional`
- `docs/CODE_REVIEW_REPORT.md:40` -> `Tauri desktop | npm run dev:tauri`

Assessment
- This is documentation drift, not necessarily code-architecture drift.
- It is already accounted for by the new authority docs, but these files remain a confusion risk.

Recommended follow-through
- Reclassify or annotate these docs as historical/non-authoritative, or update them to match the current handoff.
- Preserve quoted stale lines only as evidence, not as restated runnable guidance.

### 2. Native support surfaces are first-class build inputs

Finding
- `src_penta-core/` and `libs/daiw/` are not merely archival remnants; they are still wired directly into the root build.

Evidence
- `CMakeLists.txt:179` -> `add_subdirectory(src_penta-core)`
- `CMakeLists.txt:253` -> `add_subdirectory(libs/daiw)`
- `src_penta-core/CMakeLists.txt:60-62` explicitly says `src_penta-core/` is canonical for `penta::` namespace symbols in the current consolidation
- `libs/daiw/CMakeLists.txt:5` documents dependency chain `daiw → penta → kelly`

Assessment
- Pass B classification is correct: these are native support/code-organization surfaces, not separate semantic or persistence authority layers.
- But they are operationally important and cannot be treated as dead code or passive legacy without more consolidation work.

Recommended follow-through
- Keep them classified as support surfaces in docs.
- When refactoring native code, require reviewers to check build wiring as well as architectural authority.

### 3. KellyCore target boundaries are enforced by exclusions, not clean source partitioning

Finding
- The main native target is built from a broad recursive `src/*.cpp` / `src/*.mm` collection and then narrowed by a long list of exclusion rules.

Evidence
- `CMakeLists.txt:258-261` -> `file(GLOB_RECURSE KELLY_CORE_SOURCES ... src/*.cpp ... src/*.mm)`
- `CMakeLists.txt:263-290` -> multiple exclusion filters for plugin files, daiw files, project file, Python bridge variants, and other duplicates
- `CMakeLists.txt:292-303` -> additional ODR-oriented exclusions, including duplicate UI look-and-feel sources

Assessment
- This is the strongest module-boundary drift risk.
- The approved architecture now defines clearer module ownership than the target graph currently enforces.
- The build still relies on exclusion discipline and comments to keep boundaries coherent.

Recommended follow-through
- Future cleanup should migrate from exclusion-heavy globbing toward more explicit target/source partitioning.
- Treat build-target reshaping as a human-reviewed architectural cleanup, not a casual agent refactor.

### 4. `src/ui/` is not yet mechanically isolated as a plugin-runtime presentation surface

Finding
- Pass B places `src/ui/` under plugin/runtime authority as presentation.
- The current root build excludes `src/plugin/` from KellyCore but does not similarly isolate `src/ui/`.

Evidence
- `CMakeLists.txt:265` excludes `/src/plugin/`
- No equivalent exclusion for the whole `/src/ui/` subtree appears in the same boundary section
- `CMakeLists.txt:300-303` references duplicate handling inside `src/ui/`, implying those files are still part of KellyCore assembly

Assessment
- This is not necessarily a violation of the architecture, but it means the module boundary exists primarily as policy/doc authority rather than enforced target isolation.
- Future contributors may incorrectly infer that all `src/ui/` code belongs to a general core target rather than the plugin-runtime shell.

Recommended follow-through
- Keep this as an explicit drift item.
- If/when target structure is cleaned up, `src/ui/` should be reviewed together with plugin/runtime ownership boundaries.

### 5. Persistence authority looks aligned at surface level

Finding
- The repo still exposes persistence through plugin/runtime-facing surfaces, which is consistent with Pass G.

Evidence
- `src/plugin/PluginState.h` contains `saveState`, `loadState`, `savePreset`, `applyPreset`, and `writeAutosaveSnapshot`
- Earlier inspection also identified `src/project/ProjectFile.cpp` as a project persistence surface

Assessment
- No obvious repo-level drift was found here in this audit.
- This area remains high-risk and human-owned, but the top-level ownership story appears aligned.

### 6. Canonical docs now correctly outrank older narratives

Finding
- The new authority docs now explicitly mark older Tauri-centered narratives as historical when they conflict.

Evidence
- `docs/ARCHITECTURE.md` now states the canonical handoff covers passes A through G and notes older Tauri-centered docs are historical.
- `docs/REPO_MODULE_MAP.md` explicitly states historical docs are not authority when they conflict with the new handoff set.

Assessment
- This is the main containment mechanism for existing drift.
- It lowers the risk of immediate architectural confusion, but only if future work actually consults the new docs first.

## Risk ranking

Highest risk
1. exclusion-based native target assembly obscuring module boundaries
2. historical Tauri-centered docs being mistaken for current authority

Medium risk
3. `src/ui/` not being mechanically isolated to match the Pass B module map
4. support surfaces (`src_penta-core/`, `libs/daiw/`, `include/penta/`, `include/prrot/`) being mistaken for separate truth-owning layers

Lower immediate risk
5. persistence surface ownership at repo level appears aligned, though still sensitive

## Recommended next implementation/governance steps

1. Annotate or update historical Tauri-era docs that still present themselves as current architecture.
2. Add lightweight architecture guardrails so CI/review catches missing authority docs or partial handoff regressions.
3. Plan a human-reviewed build-graph cleanup to reduce exclusion-heavy source globbing in KellyCore.
4. When touching `src/ui/`, explicitly review whether the change belongs to plugin/runtime presentation versus general native core.

## Non-goals of this audit

This audit does not claim:
- that all runtime ownership code has been revalidated file-by-file
- that all persistence code fully matches the approved architecture in implementation detail
- that the current build graph is wrong in every mixed-target case

It only identifies the highest-signal drift and ambiguity surfaces relative to the newly approved handoff.
