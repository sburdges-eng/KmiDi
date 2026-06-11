# xformat — pre-merge verification gate

`xformat` is a guardrail against agentic-coding regressions. Before a branch is
pushed or merged, it runs the checks that catch the failure modes an autonomous
agent is most likely to introduce — lint/type regressions, broken tests, schema
drift, and edits that stray outside the task's declared scope.

It exists because automated agents *do* drift: the 2026-05-23 swarm run produced
feature branches off a stale base, two of which re-implemented code that had
already landed on `main` (see `docs/MERGE_PLAN_2026-05-23.md`). `xformat` is the
gate that would have flagged that before the work was proposed.

## Checks

Every check is scoped to *what actually changed* and SKIPs when irrelevant, so
the gate stays fast on small diffs.

| Check | Runs when | What it does |
|-------|-----------|--------------|
| `scope` | `--scope` declared | Changed files must match an allowed glob; out-of-scope edits FAIL |
| `lint` | `.py` changed | `flake8 --max-line-length 100` on changed Python files |
| `typecheck` | `.ts`/`.tsx` changed | `npx tsc --noEmit` |
| `tests` | test files / mapped modules changed | `pytest` on changed + affected tests (`--full` = whole suite) |
| `rust` | `engine/intent_ir/**.rs` changed | `cargo check` (`--full` = `cargo test`) |
| `schema` | `shared_schemas/**` or `engine_api/schema*` changed | regenerates via `sync_entities.py`, then git-diffs the generated mirrors; drift = FAIL (tree is restored either way) |
| `gitnexus` | always | `gitnexus status` freshness — WARN only |

Exit code: **0** when every check is PASS / SKIP / WARN, **1** when any FAILs.

### GitNexus caveat

`gitnexus impact` and `gitnexus detect_changes` are **MCP tools with no CLI**, so
they cannot run inside a shell hook. They remain an **agent-time obligation**
(see `CLAUDE.md` → "Always Do"). The `gitnexus` check here only flags a *stale
index*, which would make any agent-side impact analysis unreliable.

## Usage

```bash
# Diff against origin/main (default), affected checks only
python3 scripts/xformat.py

# Enforce a scope allowlist (mirrors a subagent's declared file boundary)
python3 scripts/xformat.py --scope 'music_brain/latent/**' --scope 'tests/unit/**'

# Include uncommitted + untracked files, not just base...HEAD
python3 scripts/xformat.py --include-dirty

# Whole pytest suite / cargo test instead of only affected targets
python3 scripts/xformat.py --full

# Diff against a different base
python3 scripts/xformat.py --base origin/feat/some-branch
```

## Install as a pre-push hook (opt-in)

The hooks in `scripts/hooks/` chain Git LFS (preserved) and the xformat gate.
Enable them per-clone:

```bash
git config core.hooksPath scripts/hooks      # enable
git config --unset core.hooksPath            # disable
```

Bypass for a single push (e.g. a WIP branch):

```bash
XFORMAT_SKIP=1 git push      # or: git push --no-verify
```

`scripts/hooks/` also ships forwarding stubs for `post-commit`, `post-merge`, and
`post-checkout` so that redirecting `core.hooksPath` does not disable the Git LFS
hooks the repo relies on.
