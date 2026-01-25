#!/usr/bin/env python3
"""
Find and catalog all code files across the MacBook Pro.
Analyzes file metadata, imports, and complexity.
"""

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Code file extensions
CODE_EXTENSIONS = {
    "python": {".py", ".pyx", ".pyi"},
    "cpp": {".cpp", ".hpp", ".h", ".cc", ".cxx", ".c", ".hxx"},
    "javascript": {".js", ".jsx"},
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

ALL_CODE_EXTENSIONS = set()
for exts in CODE_EXTENSIONS.values():
    ALL_CODE_EXTENSIONS.update(exts)

# Keywords to identify valuable code
KMDI_KEYWORDS = {"music_brain", "penta_core", "kelly", "kmidi", "midi", "audio"}
ML_KEYWORDS = {
    "training",
    "inference",
    "model",
    "neural",
    "tensor",
    "pytorch",
    "tensorflow",
    "keras",
}
MUSIC_KEYWORDS = {"midi", "audio", "harmony", "melody", "chord", "note", "scale", "tempo", "rhythm"}

EXCLUDE_DIRS = {
    "Library",
    "System",
    "Applications",
    "node_modules",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".venv",
    "venv",
    "env",
    "build",
    "dist",
}


def should_exclude(path: Path) -> bool:
    """Check if path should be excluded."""
    parts = path.parts
    for exclude in EXCLUDE_DIRS:
        if exclude in parts:
            return True
    return False


def get_file_language(file_path: Path) -> str:
    """Determine programming language from file extension."""
    ext = file_path.suffix.lower()
    for lang, exts in CODE_EXTENSIONS.items():
        if ext in exts:
            return lang
    return "unknown"


def analyze_python_file(file_path: Path) -> Dict:
    """Analyze Python file for imports, classes, functions."""
    analysis = {
        "imports": [],
        "classes": [],
        "functions": [],
        "has_kmidi_imports": False,
        "has_ml_keywords": False,
        "has_music_keywords": False,
        "todo_count": 0,
        "fixme_count": 0,
    }

    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")

        # Find imports
        import_pattern = r"^(?:from\s+(\S+)\s+)?import\s+(\S+)"
        for line in content.split("\n"):
            match = re.match(import_pattern, line.strip())
            if match:
                module = match.group(1) or match.group(2)
                analysis["imports"].append(module)
                if any(kw in module.lower() for kw in KMDI_KEYWORDS):
                    analysis["has_kmidi_imports"] = True

        # Find classes
        class_pattern = r"^class\s+(\w+)"
        for line in content.split("\n"):
            match = re.match(class_pattern, line.strip())
            if match:
                analysis["classes"].append(match.group(1))

        # Find functions
        func_pattern = r"^def\s+(\w+)"
        for line in content.split("\n"):
            match = re.match(func_pattern, line.strip())
            if match:
                analysis["functions"].append(match.group(1))

        # Count TODOs and FIXMEs
        analysis["todo_count"] = content.lower().count("todo")
        analysis["fixme_count"] = content.lower().count("fixme")

        # Check for keywords
        content_lower = content.lower()
        analysis["has_ml_keywords"] = any(kw in content_lower for kw in ML_KEYWORDS)
        analysis["has_music_keywords"] = any(kw in content_lower for kw in MUSIC_KEYWORDS)

    except Exception:
        pass

    return analysis


def analyze_cpp_file(file_path: Path) -> Dict:
    """Analyze C++ file for includes, classes, functions."""
    analysis = {
        "includes": [],
        "classes": [],
        "has_kmidi_includes": False,
        "has_ml_keywords": False,
        "has_music_keywords": False,
        "todo_count": 0,
        "fixme_count": 0,
    }

    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")

        # Find includes
        include_pattern = r'#include\s+[<"]([^>"]+)[>"]'
        for match in re.finditer(include_pattern, content):
            include_path = match.group(1)
            analysis["includes"].append(include_path)
            if any(kw in include_path.lower() for kw in KMDI_KEYWORDS):
                analysis["has_kmidi_includes"] = True

        # Find classes
        class_pattern = r"class\s+(\w+)"
        for match in re.finditer(class_pattern, content):
            analysis["classes"].append(match.group(1))

        # Count TODOs
        analysis["todo_count"] = content.lower().count("todo")
        analysis["fixme_count"] = content.lower().count("fixme")

        # Check keywords
        content_lower = content.lower()
        analysis["has_ml_keywords"] = any(kw in content_lower for kw in ML_KEYWORDS)
        analysis["has_music_keywords"] = any(kw in content_lower for kw in MUSIC_KEYWORDS)

    except Exception:
        pass

    return analysis


def analyze_file(file_path: Path) -> Dict:
    """Analyze a code file and return metadata."""
    try:
        stat = file_path.stat()
        language = get_file_language(file_path)

        file_info = {
            "path": str(file_path),
            "name": file_path.name,
            "language": language,
            "size": stat.st_size,
            "size_kb": round(stat.st_size / 1024, 2),
            "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "directory": str(file_path.parent),
        }

        # Language-specific analysis
        if language == "python":
            analysis = analyze_python_file(file_path)
            file_info.update(analysis)
        elif language == "cpp":
            analysis = analyze_cpp_file(file_path)
            file_info.update(analysis)

        return file_info

    except Exception:
        return None


def find_code_files(
    root: Path = Path("/Users/seanburdges"), max_files: int | None = None
) -> List[Dict]:
    """Find all code files in the directory tree. max_files=None means no limit."""
    code_files = []

    print(
        f"Scanning for code files starting from: {root}"
        + (f" (max {max_files})" if max_files else " (no limit)")
    )

    for file_path in root.rglob("*"):
        if max_files is not None and len(code_files) >= max_files:
            print(f"Reached limit of {max_files} files")
            break

        if should_exclude(file_path):
            continue

        if file_path.is_file() and file_path.suffix.lower() in ALL_CODE_EXTENSIONS:
            file_info = analyze_file(file_path)
            if file_info:
                code_files.append(file_info)

                if len(code_files) % 500 == 0 and len(code_files) > 0:
                    print(f"Found {len(code_files)} code files...")

    return code_files


def categorize_files(code_files: List[Dict]) -> Dict:
    """Categorize files by value and characteristics."""
    categories = {
        "by_language": defaultdict(int),
        "kmidi_related": [],
        "ml_related": [],
        "music_related": [],
        "large_files": [],  # > 100KB
        "recent_files": [],  # Modified in last 30 days
        "with_todos": [],
    }

    cutoff_date = datetime.now().timestamp() - (30 * 24 * 60 * 60)

    for file_info in code_files:
        # By language
        categories["by_language"][file_info["language"]] += 1

        # KmiDi related
        if (
            file_info.get("has_kmidi_imports")
            or file_info.get("has_kmidi_includes")
            or any(kw in file_info["path"].lower() for kw in KMDI_KEYWORDS)
        ):
            categories["kmidi_related"].append(file_info["path"])

        # ML related
        if file_info.get("has_ml_keywords"):
            categories["ml_related"].append(file_info["path"])

        # Music related
        if file_info.get("has_music_keywords"):
            categories["music_related"].append(file_info["path"])

        # Large files
        if file_info["size"] > 100 * 1024:
            categories["large_files"].append(
                {"path": file_info["path"], "size_kb": file_info["size_kb"]}
            )

        # Recent files
        try:
            mod_time = datetime.fromisoformat(file_info["last_modified"]).timestamp()
            if mod_time > cutoff_date:
                categories["recent_files"].append(file_info["path"])
        except:
            pass

        # Files with TODOs
        if file_info.get("todo_count", 0) > 0:
            categories["with_todos"].append(
                {"path": file_info["path"], "todo_count": file_info["todo_count"]}
            )

    return categories


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Find and catalog code files")
    parser.add_argument(
        "--max-files", type=int, default=None, help="Cap number of files (default: no limit)"
    )
    parser.add_argument(
        "--root", type=Path, default=Path("/Users/seanburdges"), help="Root directory to scan"
    )
    args = parser.parse_args()

    print("Finding all code files...")
    print("=" * 60)

    code_files = find_code_files(root=args.root, max_files=args.max_files)
    categories = categorize_files(code_files)

    # Save results
    output_dir = Path(__file__).parent.parent.parent / "docs" / "DISCOVERY"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = output_dir / "code_files_inventory.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "scan_timestamp": datetime.now().isoformat(),
                "total_files": len(code_files),
                "files": code_files,
                "categories": categories,
            },
            f,
            indent=2,
        )
    print(f"\nSaved JSON results to: {json_path}")

    # Save Markdown report
    md_path = output_dir / "code_files_inventory.md"
    with open(md_path, "w") as f:
        f.write("# Code Files Inventory\n\n")
        f.write(f"**Scan Date**: {datetime.now().isoformat()}\n\n")
        f.write(f"**Total Code Files Found**: {len(code_files):,}\n\n")

        f.write("## Files by Language\n\n")
        for lang, count in sorted(categories["by_language"].items(), key=lambda x: -x[1]):
            f.write(f"- **{lang}**: {count:,}\n")
        f.write("\n")

        f.write(f"## KmiDi-Related Files: {len(categories['kmidi_related'])}\n\n")
        for path in categories["kmidi_related"][:50]:
            f.write(f"- `{path}`\n")
        f.write("\n")

        f.write(f"## ML-Related Files: {len(categories['ml_related'])}\n\n")
        for path in categories["ml_related"][:50]:
            f.write(f"- `{path}`\n")
        f.write("\n")

        f.write(f"## Music-Related Files: {len(categories['music_related'])}\n\n")
        for path in categories["music_related"][:50]:
            f.write(f"- `{path}`\n")
        f.write("\n")

        f.write(f"## Large Files (>100KB): {len(categories['large_files'])}\n\n")
        for item in sorted(categories["large_files"], key=lambda x: -x["size_kb"])[:20]:
            f.write(f"- `{item['path']}` ({item['size_kb']} KB)\n")
        f.write("\n")

        f.write(f"## Files with TODOs: {len(categories['with_todos'])}\n\n")
        for item in sorted(categories["with_todos"], key=lambda x: -x["todo_count"])[:20]:
            f.write(f"- `{item['path']}` ({item['todo_count']} TODOs)\n")

    print(f"Saved Markdown report to: {md_path}")
    print("\n" + "=" * 60)
    print("Code file discovery complete!")
    print(f"Found {len(code_files):,} code files")


if __name__ == "__main__":
    main()
