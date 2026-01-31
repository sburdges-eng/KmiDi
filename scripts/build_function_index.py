#!/usr/bin/env python3
"""
Build a repo-wide symbol index (function/class paths) for future search and restore.

Output: TSV with path, line, kind, symbol. Use for:
  - Finding where a function/class is defined (grep symbol docs/.index/symbol_index.tsv)
  - Feeding into GIT_RESTORE_PATHWAYS (git log -S "symbol" -- path)

Run from repo root: python scripts/build_function_index.py [--canon-only]
  --canon-only  Index only KmiDi_CANON, run_brain.py, tests/, src/ (smaller, committable).

Output: docs/.index/symbol_index.tsv (and .jsonl.gz). Full run: ~400k symbols. Canon-only: ~15k.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path

# Repo root: parent of scripts/
REPO_ROOT = Path(__file__).resolve().parent.parent

# Output under docs/.index/ (create if missing)
INDEX_DIR = REPO_ROOT / "docs" / ".index"
TSV_PATH = INDEX_DIR / "symbol_index.tsv"
JSONL_GZ_PATH = INDEX_DIR / "symbol_index.jsonl.gz"
CANON_TSV_PATH = INDEX_DIR / "symbol_index_canon.tsv"

# Exclude dirs (no trailing slash; matched against path parts)
EXCLUDE_DIRS = frozenset({
    "node_modules", "__pycache__", ".git", "venv", ".venv", "build", "dist", "target",
    ".cursor", "Library", "Volumes",
})

# Exclude file patterns (suffix or name)
EXCLUDE_SUFFIXES = frozenset({".pyc", ".pyo", ".o", ".a", ".so", ".dylib", ".dll", ".exe", ".gguf", ".safetensors", ".pt", ".pth", ".bin", ".ckpt"})

# Python: def name, class Name, async def name
RE_PY_DEF = re.compile(r"^\s*(def|async\s+def|class)\s+(\w+)")
# Rust: fn name, pub fn name, struct Name, enum Name, impl Name
RE_RS = re.compile(r"^\s*(?:pub\s+)?(fn|async\s+fn|struct|enum|impl|trait)\s+(\w+)")
# C/C++: class Name, struct Name; then function name at start of line (simplified: identifier followed by ( or ::)
RE_CPP_CLASS = re.compile(r"^\s*(?:class|struct)\s+(\w+)")
RE_CPP_FUNC = re.compile(r"^\s*(?:virtual\s+|inline\s+|static\s+|explicit\s+)*(?:\w+(?:::\w+)*(?:\s*<[^>]*>)?\s+)+(\w+)\s*\(")


def should_skip(path: Path) -> bool:
    rel = path.relative_to(REPO_ROOT)
    parts = rel.parts
    if any(p in EXCLUDE_DIRS for p in parts):
        return True
    if path.suffix in EXCLUDE_SUFFIXES:
        return True
    return False


def scan_python(path: Path, rel_path: str) -> list[tuple[int, str, str]]:
    out = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for i, line in enumerate(text.splitlines(), 1):
        m = RE_PY_DEF.search(line)
        if m:
            kind = "def" if m.group(1).startswith("def") else "class"
            if "async" in m.group(1):
                kind = "async_def"
            out.append((i, kind, m.group(2)))
    return out


def scan_rust(path: Path, rel_path: str) -> list[tuple[int, str, str]]:
    out = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for i, line in enumerate(text.splitlines(), 1):
        m = RE_RS.search(line)
        if m:
            kind = m.group(1).replace(" ", "_")  # async fn -> async_fn
            out.append((i, kind, m.group(2)))
    return out


def scan_cpp(path: Path, rel_path: str) -> list[tuple[int, str, str]]:
    out = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for i, line in enumerate(text.splitlines(), 1):
        m = RE_CPP_CLASS.search(line)
        if m:
            out.append((i, "class", m.group(1)))
            continue
        m = RE_CPP_FUNC.search(line)
        if m:
            out.append((i, "function", m.group(1)))
    return out


def path_in_canon(rel_path: str) -> bool:
    return (
        rel_path.startswith("KmiDi_CANON/")
        or rel_path == "run_brain.py"
        or rel_path.startswith("tests/")
        or rel_path.startswith("src/")
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Build repo-wide symbol index for future search/restore.")
    ap.add_argument("--canon-only", action="store_true", help="Index only KmiDi_CANON, run_brain, tests, src.")
    args = ap.parse_args()
    canon_only = getattr(args, "canon_only", False)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    seen_paths: set[str] = set()
    total_symbols = 0

    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if should_skip(path):
            continue
        try:
            rel_path = path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            continue
        if canon_only and not path_in_canon(rel_path):
            continue

        entries: list[tuple[int, str, str]] = []
        suf = path.suffix.lower()
        if suf == ".py":
            entries = scan_python(path, rel_path)
        elif suf == ".rs":
            entries = scan_rust(path, rel_path)
        elif suf in (".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hh", ".hxx"):
            entries = scan_cpp(path, rel_path)

        for line_no, kind, symbol in entries:
            rows.append({"path": rel_path, "line": line_no, "kind": kind, "symbol": symbol})
            total_symbols += 1
        if entries:
            seen_paths.add(rel_path)

    # TSV (full or canon)
    out_tsv = CANON_TSV_PATH if canon_only else TSV_PATH
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("path\tline\tkind\tsymbol\n")
        for r in rows:
            f.write(f"{r['path']}\t{r['line']}\t{r['kind']}\t{r['symbol']}\n")

    # Optional: compressed JSONL (full index only when not canon-only)
    if not canon_only:
        try:
            with gzip.open(JSONL_GZ_PATH, "wt", encoding="utf-8") as z:
                for r in rows:
                    z.write(json.dumps(r, ensure_ascii=False) + "\n")
        except OSError:
            pass

    print(f"Indexed {total_symbols} symbols in {len(seen_paths)} files.", file=sys.stderr)
    print(f"TSV: {out_tsv}", file=sys.stderr)
    if not canon_only and JSONL_GZ_PATH.exists():
        print(f"JSONL.gz: {JSONL_GZ_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
