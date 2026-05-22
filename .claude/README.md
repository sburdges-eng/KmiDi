# KMiDi .claude/ assets

Multi-agent orchestration + audit + verification scaffolding, layered on top of the existing
domain agents (`audio-midi-agent`, `cpp-safety-guardian`) and per-edit verify hook.

## Layout

```
.claude/
├── agents/
│   ├── audio-midi-agent.md          # existing — audio/MIDI/DAW domain expert (sonnet)
│   ├── cpp-safety-guardian.md       # existing — C++ memory/RT/concurrency safety (opus)
│   ├── coordinator.md               # NEW — orchestrator, never auto-merges
│   ├── implementer.md               # NEW — TDD-strict per-task worker, may dispatch domain helpers
│   ├── reviewer.md                  # NEW — read-only review against KMiDi conventions
│   └── fix-it.md                    # NEW — 3 parallel hypotheses for red gates
├── commands/                        # NEW
│   ├── verify.md                    # /verify [changed|full|ts|python|cpp|rust|rt]
│   └── audit.md                     # /audit [base-branch]
├── skills/                          # NEW
│   └── audit/SKILL.md               # KMiDi-specific overlay for the global audit skill
├── verify.sh                        # existing — per-edit incremental compile (called by hook)
├── settings.local.json              # existing — credential guard, scope, lint, format, build hooks
├── launch.json                      # existing
└── worktrees/                       # orchestrator + fix-it workspaces (gitignored)
```

## Slash commands

| Command | What it does |
|---|---|
| `/audit [base]` | Security + RT/FFI/JUCE/ODR audit of branch vs `main`. Read-only. |
| `/verify [scope]` | Per-stack gates: tsc / flake8+pytest / cmake+ctest / cargo test (+ ASan if RT). |
| `/orchestrate [tasks.yaml]` | Multi-agent run — coordinator dispatches implementer + reviewer per task. Stops at `ready_to_merge`. |
| `/parallel-dispatch <n> <tasks>` | (global) Generic parallel subagent dispatcher. |
| `/dry-run-ingest <target>` | (global) — Lariat-flavored; for KMiDi you'd repurpose for dataset prep scripts. |

## Multi-agent orchestrator workflow

1. Copy `tasks.yaml.example` → `tasks.yaml`. Each task has `id`, `description`, `stack`, `acceptance_tests`, `dependencies`, `paths_touched`, optional `domain_helpers`.
2. Run `/orchestrate tasks.yaml`.
3. Coordinator creates `.claude/worktrees/<id>/`, dispatches implementer (TDD), then reviewer.
4. Watch `ORCHESTRATOR_STATUS.md` update.
5. **You** review and merge. Coordinator never merges to `main`.

### Coordinator parallelism + safety

- **Cap = 3** concurrent worktrees (JUCE submodule makes concurrent C++ builds heavy).
- **`shared_schemas/`** touches → serialize (regen affects TS/Rust/Python).
- **`external/JUCE/`**, **`KmiDi_FINAL/`**, **`KmiDi_PROJECT/`** → block (vendored / legacy paths).
- **No retry on red** — surface and stop. User runs `/orchestrate retry <id>` if desired.

### Implementer ↔ domain agents

The implementer may dispatch `cpp-safety-guardian` for C++ files touching RT/FFI before commit, and `audio-midi-agent` for audio/MIDI domain logic. It does **not** dispatch them for trivial fixes (typos, comments). The orchestration layer is complementary to the domain agents — not a replacement.

### Fix-It

When a gate goes red, `/fix-it` (or invoking the agent) runs **3 parallel hypothesis subagents** in scratch worktrees `.claude/worktrees/fix-H<n>`. Each proposes a different root cause + minimal fix, runs the gate, and reports outcome. **You pick which to merge.** Never auto-applied.

## Audit overlay (KMiDi-specific categories)

Beyond the global `audit` skill, the KMiDi overlay (`.claude/skills/audit/SKILL.md`) checks:

- **RT / audio thread**: heap alloc / lock / blocking I/O / `throw` on `processBlock` paths.
- **FFI / KellyFFI**: ownership documentation, no JUCE-link in KellyFFI consumers, allocator pairing across the C-ABI.
- **JUCE / Qt link discipline**: PRIVATE-only on `KellyFFI` target, no double-JUCE in build graph.
- **ODR / static-init / allocator** mismatches across translation units.
- **Sanitizer exclusivity**: ASan and TSan never both ON.
- **Python / GIL**: pybind11 GIL acquire/release correctness; no `--timeout` to pytest.
- **Schema sync drift**: `shared_schemas/` edits paired with `sync_entities.py` regen; generated files not hand-edited.
- **Dataset / weight path leakage**: hardcoded `/Users/<name>/...` or `/Volumes/...`; staged audio/MIDI/`.pt`; missing `run_manifest.yaml`.
- **MIDI-CI / Core ML**: sub-4-bit production / unpinned coremltools / unvalidated ANE fallback (per 2026-03-31 watchlist).
- **App entrypoint drift**: imports of legacy `App.tsx` instead of `AppConsole`.
- **Legacy paths**: new files in `KmiDi_FINAL/`, `KmiDi_PROJECT/`, `.worktrees/integration-finalize/KmiDi_FINAL/`.

## Existing infrastructure that this layered on top of

- **Per-edit verify hook**: `.claude/verify.sh` (incremental ninja build) + the PostToolUse hooks in `settings.local.json` (clang-format / prettier / pytest / tsc / flake8 / cmake target build).
- **Workspace-scaffold hooks**: credential guard, scope enforcement, lint-on-write.
- **Anchor docs**: `AGENTS.md` § *Native safety, FFI ownership, and verification map* — single source of truth for FFI/RT rules. The audit overlay is consistent with it; if they conflict, AGENTS.md wins.

## Why no auto-merge / auto-stash / auto-fix

Auto-merging worktrees to `main` is destructive shared-state without review.
Auto-stashing the user's working tree to "fix" a hook failure can lose in-progress work.
Auto-applying lint fixes on safety-critical C++ hides issues you should see.

The orchestrator and fix-it agent stop at "here are the options." You pick.

## Verify your setup

```bash
# Confirm new files
find .claude -type f | sort

# Try a dry-run audit (no diff = no findings)
# /audit main

# Try /verify on the current state
# /verify changed

# Read the example manifest
cat tasks.yaml.example
```
