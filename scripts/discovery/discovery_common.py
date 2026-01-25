#!/usr/bin/env python3
"""
Shared utilities for comprehensive file discovery.

Design goals:
- Single-pass scanning (avoid Path.rglob per extension)
- Conservative content reading (small text files only)
- Prune noisy/system directories while still allowing cloud-storage symlinks
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple


HOME_DEFAULT = Path("/Users/seanburdges")


CODE_EXTENSIONS: Dict[str, Set[str]] = {
    "python": {".py", ".pyx", ".pyi", ".pyw"},
    "cpp": {".cpp", ".hpp", ".h", ".cc", ".cxx", ".c", ".hxx"},
    "javascript": {".js", ".jsx", ".mjs"},
    "typescript": {".ts", ".tsx"},
    "rust": {".rs"},
    "go": {".go"},
    "java": {".java"},
    "swift": {".swift"},
    "kotlin": {".kt"},
    "scala": {".scala"},
    "ruby": {".rb"},
    "php": {".php"},
    "shell": {".sh", ".bash", ".zsh"},
}

DOC_EXTENSIONS: Dict[str, Set[str]] = {
    "markdown": {".md", ".markdown", ".mdown"},
    "text": {".txt", ".text"},
    "rst": {".rst"},
    "html": {".html", ".htm"},
    "pdf": {".pdf"},
    "org": {".org"},
    "wiki": {".wiki"},
}

MODEL_EXTENSIONS: Dict[str, Set[str]] = {
    "pytorch": {".pt", ".pth"},
    "tensorflow": {".pb", ".h5", ".ckpt", ".ckpt.index"},
    "onnx": {".onnx"},
    "tflite": {".tflite"},
    "coreml": {".mlmodel"},
    "pickle": {".pkl", ".pickle"},
    "numpy": {".npy", ".npz"},
}

PROJECT_MARKERS: Dict[str, Set[str]] = {
    "git": {".git"},
    "cmake": {"CMakeLists.txt"},
    "node": {"package.json"},
    "python": {"requirements.txt", "pyproject.toml", "setup.py"},
    "rust": {"Cargo.toml"},
    "go": {"go.mod"},
}


DEFAULT_PRUNE_DIR_NAMES: Set[str] = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    "__pycache__",
    "node_modules",
    "build",
    "dist",
    "target",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    ".nox",
    ".cache",
}

# Under $HOME, treat these as "system-ish" and prune by default.
DEFAULT_PRUNE_TOP_LEVEL: Set[str] = {
    "System",
    "Library",  # except allowlisted cloud storage
    ".Trash",
}


def repo_root_from_here(here: Path) -> Path:
    # scripts/discovery/<file>.py -> scripts -> repo root
    return here.resolve().parents[2]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_relpath(path: Path, home: Path) -> str:
    try:
        return str(path.resolve().relative_to(home.resolve()))
    except Exception:
        return str(path)


def classify_code(path: Path) -> Optional[str]:
    if path.name == "CMakeLists.txt":
        return "cmake"
    ext = path.suffix.lower()
    for lang, exts in CODE_EXTENSIONS.items():
        if ext in exts:
            return lang
    return None


def classify_doc(path: Path) -> Optional[str]:
    ext = path.suffix.lower()
    for doc_type, exts in DOC_EXTENSIONS.items():
        if ext in exts:
            return doc_type
    return None


def classify_model(path: Path) -> Optional[str]:
    name = path.name
    if name.endswith(".ckpt.index"):
        return "tensorflow"
    ext = path.suffix.lower()
    for model_type, exts in MODEL_EXTENSIONS.items():
        if ext in exts:
            return model_type
    return None


def is_under_any(path: Path, prefixes: Sequence[Path]) -> bool:
    path_str = str(path)
    for prefix in prefixes:
        prefix_str = str(prefix)
        if path_str == prefix_str or path_str.startswith(prefix_str + os.sep):
            return True
    return False


def default_allowlisted_prefixes(home: Path) -> List[Path]:
    # Cloud storage typically lives under Library/CloudStorage but is user data.
    return [home / "Library" / "CloudStorage"]


def should_prune_dir(
    dir_path: Path,
    *,
    home: Path,
    prune_names: Set[str],
    allowlisted_prefixes: Sequence[Path],
) -> bool:
    try:
        resolved = dir_path.resolve()
    except Exception:
        resolved = dir_path

    if is_under_any(resolved, allowlisted_prefixes):
        return False

    parts = resolved.parts
    if len(parts) >= len(home.parts) + 1 and parts[: len(home.parts)] == home.parts:
        top_level = parts[len(home.parts)]
        if top_level in DEFAULT_PRUNE_TOP_LEVEL:
            return True

    return resolved.name in prune_names


def iter_scan_roots(home: Path, *, include_cloud_symlinks: bool = True) -> List[Path]:
    roots = [home]
    allowlisted = default_allowlisted_prefixes(home)

    if include_cloud_symlinks:
        try:
            for entry in os.scandir(home):
                try:
                    if not entry.is_symlink():
                        continue
                    target = Path(entry.path).resolve()
                    if is_under_any(target, allowlisted):
                        roots.append(target)
                except OSError:
                    continue
        except OSError:
            pass

    # Deduplicate while preserving order
    out: List[Path] = []
    seen: Set[str] = set()
    for r in roots:
        key = str(r)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def iter_files(
    roots: Sequence[Path],
    *,
    home: Path,
    prune_names: Set[str] = DEFAULT_PRUNE_DIR_NAMES,
    allowlisted_prefixes: Optional[Sequence[Path]] = None,
) -> Iterator[Path]:
    allowlisted_prefixes = list(allowlisted_prefixes or default_allowlisted_prefixes(home))

    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
            dir_path = Path(dirpath)
            # Prune in-place
            kept: List[str] = []
            for name in dirnames:
                candidate = dir_path / name
                if should_prune_dir(
                    candidate, home=home, prune_names=prune_names, allowlisted_prefixes=allowlisted_prefixes
                ):
                    continue
                kept.append(name)
            dirnames[:] = kept

            for name in filenames:
                yield dir_path / name


def try_stat(path: Path) -> Optional[os.stat_result]:
    try:
        return path.stat()
    except (OSError, PermissionError):
        return None


def format_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(n)
    for u in units:
        if size < 1024.0:
            return f"{size:.1f} {u}"
        size /= 1024.0
    return f"{size:.1f} PB"


def read_text_sample(path: Path, *, max_bytes: int = 256_000) -> Optional[str]:
    st = try_stat(path)
    if st is None:
        return None
    if st.st_size == 0:
        return ""
    if st.st_size > max_bytes:
        return None
    try:
        data = path.read_bytes()
    except (OSError, PermissionError):
        return None
    if b"\x00" in data[:4096]:
        return None
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def sha256_file(path: Path, *, max_bytes: int = 5 * 1024 * 1024) -> Optional[str]:
    st = try_stat(path)
    if st is None:
        return None
    if st.st_size > max_bytes:
        return None
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except (OSError, PermissionError):
        return None


_RE_TODO = re.compile(r"\b(TODO|FIXME|HACK|XXX)\b", re.IGNORECASE)
_RE_IMPORT_JS = re.compile(r"^\s*import\s+.*?\s+from\s+['\"]([^'\"]+)['\"]", re.MULTILINE)
_RE_REQUIRE_JS = re.compile(r"require\(\s*['\"]([^'\"]+)['\"]\s*\)")
_RE_USE_RS = re.compile(r"^\s*use\s+([^;]+);", re.MULTILINE)


def extract_todos(text: str, *, limit: int = 30) -> List[str]:
    out: List[str] = []
    for line in text.splitlines():
        if _RE_TODO.search(line):
            out.append(line.strip())
        if len(out) >= limit:
            break
    return out


def extract_imports(text: str, *, language: str) -> List[str]:
    if language == "python":
        # Lightweight regex fallback; full AST parsing happens in analyze_files.py
        imports: Set[str] = set()
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("import "):
                imports.add(line.split()[1].split(",")[0])
            elif line.startswith("from "):
                parts = line.split()
                if len(parts) >= 2:
                    imports.add(parts[1])
        return sorted(imports)
    if language in {"javascript", "typescript"}:
        imports = set(_RE_IMPORT_JS.findall(text)) | set(_RE_REQUIRE_JS.findall(text))
        return sorted(imports)
    if language == "rust":
        return [m.strip() for m in _RE_USE_RS.findall(text)[:200]]
    return []


def extract_headings(text: str, *, limit: int = 50) -> List[str]:
    headings: List[str] = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            headings.append(s)
        if len(headings) >= limit:
            break
    return headings


def keyword_hits(text: str, keywords: Sequence[str]) -> List[str]:
    lowered = text.lower()
    hits = [k for k in keywords if k.lower() in lowered]
    return sorted(set(hits), key=str.lower)


KMIDI_KEYWORDS = ["kmidi", "kmi-di", "music_brain", "penta_core", "kelly", "penta-core"]
ML_KEYWORDS = [
    "train",
    "training",
    "inference",
    "model",
    "neural",
    "transformer",
    "pytorch",
    "tensorflow",
    "onnx",
    "checkpoint",
    "embedding",
]
MUSIC_KEYWORDS = ["midi", "audio", "melody", "harmony", "rhythm", "chord", "tempo", "pitch", "note", "scale"]


def value_score(
    *,
    size_bytes: int,
    mtime: datetime,
    kmidi_hits: int,
    ml_hits: int,
    music_hits: int,
    has_todos: bool,
) -> float:
    size_score = 0.0
    if size_bytes >= 200_000:
        size_score = 3.0
    elif size_bytes >= 50_000:
        size_score = 2.0
    elif size_bytes >= 10_000:
        size_score = 1.0

    days = (datetime.now(timezone.utc) - mtime).days
    recency = 0.0
    if days <= 30:
        recency = 3.0
    elif days <= 180:
        recency = 2.0
    elif days <= 365:
        recency = 1.0

    keyword = min(6.0, (kmidi_hits * 2.5) + (ml_hits * 1.0) + (music_hits * 0.75))
    todo = 0.5 if has_todos else 0.0

    return size_score + recency + keyword + todo


def value_bucket(score: float) -> str:
    if score >= 9.0:
        return "Critical"
    if score >= 6.0:
        return "High"
    if score >= 3.0:
        return "Medium"
    return "Low"

