#!/usr/bin/env python3
"""
Discover and copy KmiDi-related branches/worktrees from Cursor, Claude, Codex,
Gemini, and Copilot into experiments/, with exclusions, and write a manifest.

Plan: KmiDi Branch/Worktree Copy and Manifest.
No move/delete; copy only. Manifest is required.
"""

from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path

# Destination root (KmiDi experiments dir)
EXPERIMENTS_ROOT = Path(__file__).resolve().parent.parent / "experiments"
MANIFEST_MD = EXPERIMENTS_ROOT / "BRANCH_WORKTREE_COPY_MANIFEST.md"
MANIFEST_CSV = EXPERIMENTS_ROOT / "copy_manifest.csv"

# Max file size to copy (bytes); skip larger files
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB

# Excluded directory names (anywhere in tree)
EXCLUDED_DIRS = frozenset({
    "node_modules", "build", "dist", ".git", "__pycache__",
    "Datasets", ".cursor", ".claude", ".codex", ".gemini", ".copilot",
})

# Excluded file extensions (lowercase)
EXCLUDED_EXTENSIONS = frozenset({
    ".wav", ".mp3", ".flac", ".aiff", ".aif", ".ogg", ".m4a", ".wma",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".heic",
    ".mp4", ".mov", ".avi", ".webm",
})

# Candidate roots (under home): (tool_label, path_relative_to_home)
CANDIDATE_ROOTS = [
    ("Cursor", ".cursor/worktrees/KmiDi-1"),
    ("Claude", ".claude-worktrees"),
    ("Claude", ".claude"),
    ("Codex", ".codex"),
    ("Gemini", ".gemini"),
    ("Gemini", "Library/Application Support/Gemini"),
    ("Copilot", ".copilot"),
    ("Copilot", ".github/copilot"),
]


def _cursor_kmidi_root(home: Path) -> Path:
    return (home / ".cursor/worktrees/KmiDi-1").resolve()


def is_kmidi_related(path: Path, parent_root: Path) -> bool:
    """True if this path is a KmiDi-related branch/worktree."""
    home = Path.home()
    parent_resolved = parent_root.resolve()
    # Direct child of Cursor KmiDi-1 worktrees: all children count
    if parent_resolved == _cursor_kmidi_root(home):
        return True
    # For other roots (Claude, Codex, etc.): path must contain KmiDi markers
    path_str = path.as_posix().lower()
    for token in ("kmidi", "kelly-midi", "idaw", "dev/kmidi"):
        if token in path_str:
            return True
    if "KmiDi" in path.as_posix() or "KmiDi-1" in path.as_posix():
        return True
    return False


def discover_worktrees(home: Path) -> list[tuple[Path, str]]:
    """Discover (absolute_path, copy_name) for each KmiDi-related worktree."""
    out: list[tuple[Path, str]] = []
    seen_roots = set()

    for _tool, rel in CANDIDATE_ROOTS:
        root = home / rel
        if not root.is_dir():
            continue
        # Avoid duplicate roots (e.g. .claude twice)
        root_resolved = root.resolve()
        if root_resolved in seen_roots:
            continue
        seen_roots.add(root_resolved)

        try:
            for entry in root.iterdir():
                if not entry.is_dir():
                    continue
                if entry.name.startswith("."):
                    continue
                if is_kmidi_related(entry, root):
                    name = entry.name
                    out.append((entry.resolve(), name))
        except OSError:
            continue

    return out


def uniquify_names(entries: list[tuple[Path, str]]) -> list[tuple[Path, str]]:
    """Ensure copy names are unique; suffix _2, _3 as needed."""
    used: dict[str, int] = {}
    result: list[tuple[Path, str]] = []

    for path, name in entries:
        base = name
        if base not in used:
            used[base] = 1
            result.append((path, base))
        else:
            used[base] += 1
            result.append((path, f"{base}_{used[base]}"))
    return result


def should_skip(path: Path, name: str, is_dir: bool) -> bool:
    if is_dir:
        return name in EXCLUDED_DIRS
    suf = (path.suffix or "").lower()
    return suf in EXCLUDED_EXTENSIONS


def copy_tree(src: Path, dst: Path) -> None:
    """Copy src to dst, applying exclusions and size cap. Creates dst."""
    dst.mkdir(parents=True, exist_ok=True)
    for root, dirs, files in os.walk(src, topdown=True):
        root_path = Path(root)
        rel = root_path.relative_to(src)
        dst_dir = dst / rel
        dst_dir.mkdir(parents=True, exist_ok=True)

        # Don't descend into excluded dirs
        dirs[:] = [d for d in dirs if not should_skip(root_path / d, d, True)]

        for f in files:
            src_file = root_path / f
            if should_skip(src_file, f, False):
                continue
            try:
                size = src_file.stat().st_size
            except OSError:
                continue
            if size > MAX_FILE_BYTES:
                continue
            dst_file = dst_dir / f
            try:
                shutil.copy2(src_file, dst_file)
            except OSError:
                pass


def write_manifest(rows: list[tuple[str, str, str]]) -> None:
    """Write BRANCH_WORKTREE_COPY_MANIFEST.md and copy_manifest.csv."""
    EXPERIMENTS_ROOT.mkdir(parents=True, exist_ok=True)

    with open(MANIFEST_MD, "w", encoding="utf-8") as f:
        f.write("# Branch/Worktree Copy Manifest\n\n")
        f.write("| Original path | Destination path | Copy name |\n")
        f.write("|---------------|------------------|----------|\n")
        for orig, dest, name in rows:
            f.write(f"| {orig} | {dest} | {name} |\n")
        if not rows:
            f.write("(No copies performed.)\n")

    with open(MANIFEST_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["original_path", "destination_path", "copy_name"])
        w.writerows(rows)


def main() -> None:
    home = Path.home()
    entries = discover_worktrees(home)
    entries = uniquify_names(entries)

    rows: list[tuple[str, str, str]] = []
    for src, copy_name in entries:
        dst = EXPERIMENTS_ROOT / copy_name
        if dst.exists() and dst != src:
            # Avoid overwriting existing experiment dirs; use a unique suffix
            n = 1
            while (EXPERIMENTS_ROOT / f"{copy_name}_run{n}").exists():
                n += 1
            copy_name = f"{copy_name}_run{n}"
            dst = EXPERIMENTS_ROOT / copy_name
        try:
            copy_tree(src, dst)
        except Exception:
            continue
        rows.append((str(src), str(dst), copy_name))

    write_manifest(rows)
    print(f"Manifest written: {MANIFEST_MD}")
    print(f"CSV written: {MANIFEST_CSV}")
    print(f"Copied {len(rows)} worktrees.")


if __name__ == "__main__":
    main()
