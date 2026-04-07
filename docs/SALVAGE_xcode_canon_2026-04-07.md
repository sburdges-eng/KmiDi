# Salvage record — xcode/KmiDi/KmiDi_CANON snapshot — 2026-04-07

**Created by:** KmiDi C++ deep audit (Apr 2026)
**Purpose:** Triple-paranoia backup of the `xcode/KmiDi/KmiDi_CANON/` tree.
**Trigger:** The audit's Tree C / Group E pass identified 406 native files in `xcode/KmiDi/KmiDi_CANON/` with 50 `ACTIVE_DEVELOPMENT.md` markers, and the parent KmiDi repo's git log showed **0 commits** affecting that path. Before audit was begun, this raised concern that the work was untracked WIP.

## What was actually found

`xcode/KmiDi/` is a **nested git repository** (it has its own `.git/` directory). It is NOT registered as a submodule (`.gitmodules` only declares `external/JUCE`). The parent KmiDi repo treats `xcode/KmiDi/` as untracked, hence the explorer's "0 commits" finding.

**Inside the nested `xcode/KmiDi/.git` repo as of 2026-04-07:**
- Branch: `kmidi-companion-dev`
- Commit count: ≥11 (HEAD: `ee30673d Replace prints with logging and update cloud docs`)
- Files tracked under `KmiDi_CANON/`: 599
- Total disk size: 5.7 MB

So **the work was preserved all along** — just in a separate git repo. The parent KmiDi audit was correct that the parent had no record, but the actual content history is in the nested repo.

## Three independent snapshots created today

1. **Tar backup** (filesystem layer): `_archive/xcode-canon-snapshot-2026-04-07.tar.gz` (1.1 MB compressed). Restore with:
   ```bash
   cd /Users/seanburdges/Dev/KmiDi
   tar -xzf _archive/xcode-canon-snapshot-2026-04-07.tar.gz
   ```

2. **Parent-repo gitlink snapshot** (this commit): On the `salvage/xcode-canon-2026-04-07` branch in the parent KmiDi repo, the path `xcode/KmiDi` was added as an embedded-repo gitlink pointing to commit `ee30673d` of the nested repo's `kmidi-companion-dev` branch. This is similar to a submodule reference but without `.gitmodules` registration. To inspect:
   ```bash
   cd /Users/seanburdges/Dev/KmiDi
   git checkout salvage/xcode-canon-2026-04-07
   cd xcode/KmiDi
   git log --oneline -1   # should show ee30673d
   ```

3. **Nested-repo native history** (already existed before this audit): The 11+ commits in the nested `xcode/KmiDi/.git` repo. To inspect:
   ```bash
   cd /Users/seanburdges/Dev/KmiDi/xcode/KmiDi
   git log --oneline
   git branch -a
   ```

## What the audit found in the nested repo's content (relevant if anyone touches it again)

From Group E of the audit (the partition that walked `xcode/KmiDi/KmiDi_CANON/`), the most concerning issues were:
- ~16 JUCE Component-lifetime UAF bugs (missing `setLookAndFeel(nullptr)` and `stopTimer()` in dtors of `EmotionWorkstation`, `LyricDisplay`, `WorkstationPanel`, `MusicTheoryWorkstation`, `ScoreEntryPanel`)
- 3 file-chooser callbacks capturing `this` raw without `juce::Component::SafePointer<>`
- `MessageManager::callAsync` UAF in PluginIRInspector
- `WavetableSynth::loadWavetable` non-atomic shared_ptr swap during audio playback
- `GlottalSource::process` function-local static differentiator state shared across all instances
- `RTLogger::getLogger` non-thread-safe singleton init
- `BiometricInput` destructor deletes hardware bridges before stopping streaming thread
- `OrchestratorBridge` Python C API called without GIL from detached thread + raw `this` capture

**Critical follow-up:** Many of the same basenames exist in the active `src/ui/`, `src/plugin/`, `src/biometric/`, `src/bridge/` trees with diverged content. The same JUCE Component lifetime bugs almost certainly exist in the actually-shipped `src/ui/*.cpp` files and need their own audit pass.

## Why the salvage was done despite the work already being preserved

Belt-and-suspenders. The user invested thousands of hours; the audit's read of the situation (untracked WIP) was wrong but the cost of being paranoid was tiny (1.1 MB of tar + a one-line gitlink commit). If the nested repo is ever lost, broken, accidentally deleted, or has its `.git` directory corrupted, the tar backup remains.

## Don't accidentally delete

- `_archive/xcode-canon-snapshot-2026-04-07.tar.gz` — keep until the parent-repo / nested-repo consolidation question is decided
- `salvage/xcode-canon-2026-04-07` branch — keep as a permanent reference; merge or delete only after the user explicitly decides what to do with the nested repo
- `xcode/KmiDi/.git/` — **never delete** without first confirming the nested repo's commits are preserved elsewhere (e.g. pushed to a remote)

## Related audit artifacts

- Full audit report: `docs/CODEAUDIT_CXX_DEEP_2026Q2.md`
- Action plan: `~/.claude/plans/whimsical-stargazing-anchor.md`
- Per-pair divergence catalog: `docs/SALVAGE_CATALOG_2026Q2.md`
- Memory file: `~/.claude/projects/-Users-seanburdges/memory/project_kmidi_codeaudit_2026_q2.md`
