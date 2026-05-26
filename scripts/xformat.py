#!/usr/bin/env python3
"""xformat — pre-merge verification gate.

A guardrail against agentic-coding regressions. Before a branch is pushed or
merged, ``xformat`` runs the checks that catch the failure modes an autonomous
agent is most likely to introduce: lint/type regressions, broken tests, schema
drift between the canonical contract and its generated mirrors, and edits that
stray outside the task's declared scope.

Each check is *scoped to what actually changed* and SKIPs when irrelevant, so
the gate stays fast on small diffs:

  - scope     changed files must match the declared --scope allowlist
  - lint      flake8 on changed Python files
  - typecheck ``tsc --noEmit`` when TS/TSX changed
  - tests     pytest on changed + affected test files (--full = whole suite)
  - rust      ``cargo check`` when engine/intent_ir Rust changed
  - schema    regenerate via sync_entities.py, then git-diff the generated
              mirrors; drift = FAIL
  - gitnexus  ``gitnexus status`` freshness warning. NOTE: gitnexus impact /
              detect_changes are MCP tools with no CLI, so they cannot run from
              a shell hook — they remain an *agent-time* obligation (see
              CLAUDE.md "Always Do"). This check only flags a stale index,
              which would make any agent-side impact analysis unreliable.

Exit code: 0 when every check is PASS / SKIP / WARN; 1 when any check FAILs.

Run manually:   python3 scripts/xformat.py --base origin/main
With scope:     python3 scripts/xformat.py --scope 'music_brain/latent/**' --scope 'tests/unit/**'
Whole suite:    python3 scripts/xformat.py --full
"""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

# Generated mirrors that sync_entities.py owns. Drift in any of these means the
# committed contract is out of sync with the canonical Python schema.
SCHEMA_SOURCE_GLOBS = (
    "shared_schemas/*",
    "music_brain/engine_api/schema.py",
    "music_brain/engine_api/schema/*",
)
SCHEMA_GENERATED_PATHS = (
    "src/types/Intent.ts",
    "src/types/EmotionState.ts",
    "src/types/IntentFrame.ts",
    "engine/intent_ir/src/generated/intent.rs",
    "engine/intent_ir/src/generated/emotion.rs",
    "engine/intent_ir/src/generated/intent_frame.rs",
    "shared_schemas/CompleteSongIntentRequest.json",
    "shared_schemas/emotion_schema.json",
    "shared_schemas/intent_frame_schema.json",
    "engine/intent_ir/src/generated/mod.rs",
)

PASS, FAIL, SKIP, WARN = "PASS", "FAIL", "SKIP", "WARN"


@dataclass
class Result:
    name: str
    status: str
    detail: str = ""


def _run(cmd: Sequence[str], cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess:
    """Run *cmd*, capturing output. Never raises on non-zero exit."""
    return subprocess.run(
        list(cmd), cwd=str(cwd), capture_output=True, text=True, check=False
    )


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested directly)
# --------------------------------------------------------------------------- #
def scope_violations(files: Iterable[str], patterns: Sequence[str]) -> List[str]:
    """Return changed *files* that match none of the allowed glob *patterns*.

    Empty *patterns* means "no scope declared" — the caller treats that as a
    SKIP rather than a violation.
    """
    if not patterns:
        return []
    out: List[str] = []
    for f in files:
        if not any(fnmatch.fnmatch(f, p) for p in patterns):
            out.append(f)
    return out


def _matches_any(path: str, globs: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(path, g) for g in globs)


@dataclass
class Classification:
    py: List[str]
    ts: List[str]
    rust: List[str]
    cpp: List[str]
    schema: List[str]
    test_files: List[str]
    affected_tests: List[str]


def classify(files: Sequence[str]) -> Classification:
    """Bucket changed *files* by the check that cares about them."""
    py = [f for f in files if f.endswith(".py")]
    ts = [f for f in files if f.endswith((".ts", ".tsx"))]
    rust = [f for f in files if f.endswith(".rs") and f.startswith("engine/intent_ir/")]
    cpp = [f for f in files if f.endswith((".cpp", ".cc", ".cxx", ".h", ".hpp", ".inl"))]
    schema = [f for f in files if _matches_any(f, SCHEMA_SOURCE_GLOBS)]

    test_files = [
        f
        for f in py
        if f.startswith("tests/") and Path(f).name.startswith("test_")
    ]
    # Map a changed module to its likely test file(s) when they exist.
    # Try both the underscore-joined relative path (e.g. music_brain/latent/fusion.py
    # -> tests/unit/test_latent_fusion.py) and the bare stem fallback
    # (-> tests/unit/test_fusion.py).
    affected: List[str] = []
    for f in py:
        if f in test_files or not f.startswith("music_brain/"):
            continue
        rel = f[len("music_brain/"):]
        joined = "_".join(Path(rel).with_suffix("").parts)
        stem = Path(f).stem
        candidates = [f"tests/unit/test_{joined}.py"]
        if joined != stem:
            candidates.append(f"tests/unit/test_{stem}.py")
        for candidate in candidates:
            if (REPO_ROOT / candidate).exists() and candidate not in affected:
                affected.append(candidate)
    return Classification(py, ts, rust, cpp, schema, test_files, affected)


# --------------------------------------------------------------------------- #
# Git plumbing
# --------------------------------------------------------------------------- #
def changed_files(base: str, include_dirty: bool) -> tuple[List[str], bool]:
    """Files changed relative to *base* (merge-base diff), plus optionally the
    uncommitted working-tree + untracked files.

    Returns ``(sorted_files, base_ok)`` where *base_ok* is ``False`` when the
    diff against *base* failed (e.g. ref not fetched).
    """
    files: set[str] = set()
    base_ok = True
    cp = _run(["git", "diff", "--name-only", f"{base}...HEAD"])
    if cp.returncode == 0:
        files.update(line for line in cp.stdout.splitlines() if line.strip())
    else:
        base_ok = False
    if include_dirty:
        for args in (["git", "diff", "--name-only"], ["git", "diff", "--name-only", "--cached"]):
            cp = _run(args)
            files.update(line for line in cp.stdout.splitlines() if line.strip())
        cp = _run(["git", "ls-files", "--others", "--exclude-standard"])
        files.update(line for line in cp.stdout.splitlines() if line.strip())
    return sorted(files), base_ok


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def check_scope(files: Sequence[str], patterns: Sequence[str]) -> Result:
    if not patterns:
        return Result("scope", SKIP, "no --scope declared")
    bad = scope_violations(files, patterns)
    if bad:
        shown = ", ".join(bad[:5]) + (" …" if len(bad) > 5 else "")
        return Result("scope", FAIL, f"{len(bad)} file(s) outside scope: {shown}")
    return Result("scope", PASS, f"{len(files)} file(s) within scope")


def check_lint(cls: Classification) -> Result:
    if not cls.py:
        return Result("lint", SKIP, "no Python changes")
    cp = _run(["python3", "-m", "flake8", "--max-line-length", "100", *cls.py])
    if cp.returncode != 0:
        n = len([ln for ln in cp.stdout.splitlines() if ln.strip()])
        return Result("lint", FAIL, f"{n} flake8 finding(s)\n{cp.stdout.strip()}")
    return Result("lint", PASS, f"flake8 clean on {len(cls.py)} file(s)")


def check_typecheck(cls: Classification) -> Result:
    if not cls.ts:
        return Result("typecheck", SKIP, "no TS/TSX changes")
    cp = _run(["npx", "tsc", "--noEmit"])
    if cp.returncode != 0:
        return Result("typecheck", FAIL, (cp.stdout or cp.stderr).strip()[-2000:])
    return Result("typecheck", PASS, "tsc --noEmit clean")


def check_tests(cls: Classification, full: bool) -> Result:
    if full:
        targets = ["tests/"]
    else:
        targets = sorted(set(cls.test_files) | set(cls.affected_tests))
    if not targets:
        return Result("tests", SKIP, "no test files changed/affected")
    cp = _run(["python3", "-m", "pytest", *targets, "-q", "-x"])
    if cp.returncode != 0:
        tail = (cp.stdout or cp.stderr).strip().splitlines()[-25:]
        return Result("tests", FAIL, "\n".join(tail))
    return Result("tests", PASS, f"pytest green ({len(targets)} target(s))")


def check_rust(cls: Classification, full: bool) -> Result:
    if not cls.rust:
        return Result("rust", SKIP, "no engine/intent_ir Rust changes")
    crate = REPO_ROOT / "engine" / "intent_ir"
    cmd = ["cargo", "test"] if full else ["cargo", "check"]
    cp = _run(cmd, cwd=crate)
    if cp.returncode != 0:
        return Result("rust", FAIL, (cp.stderr or cp.stdout).strip()[-1500:])
    return Result("rust", PASS, f"{' '.join(cmd)} clean")


def check_schema_drift(cls: Classification) -> Result:
    if not cls.schema:
        return Result("schema", SKIP, "no schema sources changed")
    gen = _run(["python3", "scripts/sync_entities.py"])
    if gen.returncode != 0:
        msg = (gen.stderr or gen.stdout).strip()[-1200:]
        return Result("schema", FAIL, "sync_entities.py failed:\n" + msg)
    existing = [p for p in SCHEMA_GENERATED_PATHS if (REPO_ROOT / p).exists()]
    diff = _run(["git", "diff", "--name-only", "--", *existing])
    drifted = [ln for ln in diff.stdout.splitlines() if ln.strip()]
    if drifted:
        # Restore so the gate leaves the tree as it found it.
        _run(["git", "checkout", "--", *drifted])
        return Result(
            "schema",
            FAIL,
            "generated mirrors drifted from schema — run scripts/sync_entities.py and commit:\n  "
            + "\n  ".join(drifted),
        )
    return Result("schema", PASS, "generated mirrors in sync")


def check_gitnexus() -> Result:
    cp = _run(["npx", "gitnexus", "status"])
    out = (cp.stdout + cp.stderr).lower()
    if cp.returncode != 0:
        return Result("gitnexus", WARN, "gitnexus status unavailable (index/CLI missing)")
    if "stale" in out or "out of date" in out or "out-of-date" in out:
        return Result("gitnexus", WARN,
                      "index STALE — run `npx gitnexus analyze`; "
                      "agent impact analysis unreliable until then")
    return Result("gitnexus", PASS,
                  "index fresh (run gitnexus_impact/detect_changes as agent-time steps)")


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
_GLYPH = {PASS: "✓", FAIL: "✗", SKIP: "·", WARN: "!"}


def render(results: Sequence[Result]) -> str:
    width = max((len(r.name) for r in results), default=4)
    lines = ["", "  xformat — pre-merge verification gate", ""]
    for r in results:
        head = r.detail.splitlines()[0] if r.detail else ""
        lines.append(f"  {_GLYPH.get(r.status, '?')} {r.status:<4} {r.name:<{width}}  {head}")
        extra = r.detail.splitlines()[1:] if r.detail else []
        for ln in extra:
            lines.append(f"        {ln}")
    failed = [r.name for r in results if r.status == FAIL]
    lines.append("")
    lines.append("  RESULT: " + ("FAIL → " + ", ".join(failed) if failed else "PASS"))
    lines.append("")
    return "\n".join(lines)


def run_gate(base: str, scope: Sequence[str], include_dirty: bool, full: bool) -> int:
    files, base_ok = changed_files(base, include_dirty)
    cls = classify(files)
    results: list[Result] = []
    if not base_ok:
        results.append(Result(
            "base", FAIL,
            f"git diff against '{base}' failed — is the ref fetched?",
        ))
    results += [
        check_scope(files, scope),
        check_lint(cls),
        check_typecheck(cls),
        check_tests(cls, full),
        check_rust(cls, full),
        check_schema_drift(cls),
        check_gitnexus(),
    ]
    sys.stdout.write(render(results))
    return 1 if any(r.status == FAIL for r in results) else 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="xformat", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="origin/main",
                    help="ref to diff against (default: origin/main)")
    ap.add_argument("--scope", action="append", default=[], metavar="GLOB",
                    help="allowed path glob; repeatable. Files outside all globs FAIL scope.")
    ap.add_argument("--include-dirty", action="store_true",
                    help="also check uncommitted + untracked files, not just base...HEAD")
    ap.add_argument("--full", action="store_true",
                    help="run the whole pytest suite / cargo test instead of only affected targets")
    args = ap.parse_args(argv)
    return run_gate(args.base, args.scope, args.include_dirty, args.full)


if __name__ == "__main__":
    raise SystemExit(main())
