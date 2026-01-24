#!/usr/bin/env python3
"""
Comprehensive directory scanner for MacBook Pro file discovery.
Scans user directories and identifies project-like structures, large files, and valuable content.
"""

import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

# System directories to exclude
EXCLUDE_DIRS = {
    "Library",
    "System",
    "Applications",
    "SystemVolumes",
    ".Trash",
    ".Trashes",
    ".TemporaryItems",
    ".DocumentRevisions-V100",
    "node_modules",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".venv",
    "venv",
    "env",
    ".env",
    "build",
    "dist",
    ".build",
}

# Project markers
PROJECT_MARKERS = {
    ".git": "git_repo",
    "CMakeLists.txt": "cmake",
    "package.json": "nodejs",
    "requirements.txt": "python",
    "pyproject.toml": "python",
    "Cargo.toml": "rust",
    "go.mod": "go",
    "pom.xml": "maven",
    "build.gradle": "gradle",
    "Makefile": "make",
    "README.md": "documentation",
    "setup.py": "python",
    "setup.cfg": "python",
}


def should_exclude(path: Path) -> bool:
    """Check if directory should be excluded from scanning."""
    parts = path.parts
    for exclude in EXCLUDE_DIRS:
        if exclude in parts:
            return True
    return False


def scan_directory(root: Path, max_depth: int = 10, current_depth: int = 0) -> Dict:
    """Recursively scan directory structure."""
    if current_depth > max_depth or should_exclude(root):
        return {}

    if not root.exists() or not root.is_dir():
        return {}

    try:
        stats = {
            "path": str(root),
            "depth": current_depth,
            "directories": 0,
            "files": 0,
            "total_size": 0,
            "large_files": [],  # Files > 10MB
            "project_markers": [],
            "code_files": 0,
            "doc_files": 0,
            "model_files": 0,
            "last_modified": None,
            "subdirectories": [],
        }

        items = list(root.iterdir())
        dirs = [d for d in items if d.is_dir()]
        files = [f for f in items if f.is_file()]

        stats["directories"] = len(dirs)
        stats["files"] = len(files)

        # Check for project markers
        for marker, marker_type in PROJECT_MARKERS.items():
            if (root / marker).exists():
                stats["project_markers"].append({"marker": marker, "type": marker_type})

        # Analyze files
        code_extensions = {
            ".py",
            ".pyx",
            ".pyi",
            ".cpp",
            ".hpp",
            ".h",
            ".cc",
            ".cxx",
            ".js",
            ".ts",
            ".jsx",
            ".tsx",
            ".rs",
            ".go",
            ".java",
            ".swift",
            ".kt",
            ".scala",
            ".rb",
            ".c",
            ".hxx",
        }
        doc_extensions = {".md", ".markdown", ".txt", ".rst", ".html", ".htm", ".pdf", ".org"}
        model_extensions = {
            ".pt",
            ".pth",
            ".pb",
            ".h5",
            ".ckpt",
            ".onnx",
            ".tflite",
            ".mlmodel",
            ".pkl",
            ".pickle",
        }

        latest_mtime = 0
        for file in files:
            try:
                file_stat = file.stat()
                size = file_stat.st_size
                stats["total_size"] += size
                mtime = file_stat.st_mtime
                latest_mtime = max(latest_mtime, mtime)

                ext = file.suffix.lower()
                if ext in code_extensions:
                    stats["code_files"] += 1
                elif ext in doc_extensions:
                    stats["doc_files"] += 1
                elif ext in model_extensions:
                    stats["model_files"] += 1

                if size > 10 * 1024 * 1024:  # > 10MB
                    stats["large_files"].append(
                        {"path": str(file), "size": size, "size_mb": round(size / (1024 * 1024), 2)}
                    )
            except (OSError, PermissionError):
                continue

        if latest_mtime > 0:
            stats["last_modified"] = datetime.fromtimestamp(latest_mtime).isoformat()

        # Recursively scan subdirectories (limit depth for performance)
        if current_depth < 3:  # Only deep scan top 3 levels
            for subdir in dirs:
                if not should_exclude(subdir):
                    try:
                        sub_stats = scan_directory(subdir, max_depth, current_depth + 1)
                        if sub_stats:
                            stats["subdirectories"].append(sub_stats)
                    except (OSError, PermissionError):
                        continue

        # Only return if directory has interesting content
        if (
            stats["files"] > 0
            or stats["project_markers"]
            or stats["code_files"] > 0
            or stats["large_files"]
        ):
            return stats

        return {}

    except (OSError, PermissionError) as e:
        return {}


def scan_user_directories(user_home: str = "/Users/seanburdges") -> Dict:
    """Scan all top-level directories in user home."""
    user_path = Path(user_home)
    if not user_path.exists():
        return {}

    results = {
        "scan_timestamp": datetime.now().isoformat(),
        "user_home": user_home,
        "directories": [],
    }

    # Get top-level directories
    try:
        top_level_dirs = [
            d for d in user_path.iterdir() if d.is_dir() and not d.name.startswith(".")
        ]

        print(f"Scanning {len(top_level_dirs)} top-level directories...")

        for directory in sorted(top_level_dirs):
            if should_exclude(directory):
                continue

            print(f"Scanning: {directory.name}")
            stats = scan_directory(directory, max_depth=5)
            if stats:
                results["directories"].append(stats)

    except (OSError, PermissionError) as e:
        print(f"Error scanning {user_home}: {e}")

    return results


def generate_summary(scan_results: Dict) -> Dict:
    """Generate summary statistics from scan results."""
    summary = {
        "total_directories_scanned": len(scan_results.get("directories", [])),
        "total_files": 0,
        "total_size_gb": 0,
        "project_directories": 0,
        "code_files": 0,
        "doc_files": 0,
        "model_files": 0,
        "large_files_count": 0,
        "directories_by_type": defaultdict(int),
    }

    def process_directory(stats: Dict):
        summary["total_files"] += stats.get("files", 0)
        summary["total_size_gb"] += stats.get("total_size", 0) / (1024**3)
        summary["code_files"] += stats.get("code_files", 0)
        summary["doc_files"] += stats.get("doc_files", 0)
        summary["model_files"] += stats.get("model_files", 0)
        summary["large_files_count"] += len(stats.get("large_files", []))

        if stats.get("project_markers"):
            summary["project_directories"] += 1
            for marker in stats["project_markers"]:
                summary["directories_by_type"][marker["type"]] += 1

        for subdir in stats.get("subdirectories", []):
            process_directory(subdir)

    for directory in scan_results.get("directories", []):
        process_directory(directory)

    summary["total_size_gb"] = round(summary["total_size_gb"], 2)
    summary["directories_by_type"] = dict(summary["directories_by_type"])

    return summary


def main():
    """Main execution function."""
    print("Starting comprehensive directory scan...")
    print("=" * 60)

    scan_results = scan_user_directories()
    summary = generate_summary(scan_results)

    # Save results
    output_dir = Path(__file__).parent.parent.parent / "docs" / "DISCOVERY"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = output_dir / "directory_map.json"
    with open(json_path, "w") as f:
        json.dump(scan_results, f, indent=2)
    print(f"\nSaved JSON results to: {json_path}")

    # Save Markdown report
    md_path = output_dir / "directory_map.md"
    with open(md_path, "w") as f:
        f.write("# Directory Map\n\n")
        f.write(f"**Scan Date**: {scan_results['scan_timestamp']}\n\n")
        f.write("## Summary\n\n")
        f.write(f"- **Total Directories Scanned**: {summary['total_directories_scanned']}\n")
        f.write(f"- **Total Files**: {summary['total_files']:,}\n")
        f.write(f"- **Total Size**: {summary['total_size_gb']:.2f} GB\n")
        f.write(f"- **Project Directories**: {summary['project_directories']}\n")
        f.write(f"- **Code Files**: {summary['code_files']:,}\n")
        f.write(f"- **Documentation Files**: {summary['doc_files']:,}\n")
        f.write(f"- **Model Files**: {summary['model_files']:,}\n")
        f.write(f"- **Large Files (>10MB)**: {summary['large_files_count']}\n\n")

        if summary["directories_by_type"]:
            f.write("## Project Types\n\n")
            for ptype, count in sorted(summary["directories_by_type"].items()):
                f.write(f"- **{ptype}**: {count}\n")
            f.write("\n")

        f.write("## Directory Details\n\n")

        def write_directory(stats: Dict, indent: int = 0):
            prefix = "  " * indent
            f.write(f"{prefix}### {Path(stats['path']).name}\n\n")
            f.write(f"{prefix}- **Path**: `{stats['path']}`\n")
            f.write(f"{prefix}- **Files**: {stats['files']:,}\n")
            f.write(f"{prefix}- **Size**: {stats['total_size'] / (1024**3):.2f} GB\n")

            if stats.get("project_markers"):
                f.write(f"{prefix}- **Project Markers**: ")
                markers = [m["marker"] for m in stats["project_markers"]]
                f.write(", ".join(markers) + "\n")

            if stats.get("code_files", 0) > 0:
                f.write(f"{prefix}- **Code Files**: {stats['code_files']}\n")
            if stats.get("doc_files", 0) > 0:
                f.write(f"{prefix}- **Doc Files**: {stats['doc_files']}\n")
            if stats.get("model_files", 0) > 0:
                f.write(f"{prefix}- **Model Files**: {stats['model_files']}\n")

            if stats.get("large_files"):
                f.write(f"{prefix}- **Large Files**:\n")
                for lf in stats["large_files"][:10]:  # Limit to 10
                    f.write(f"{prefix}  - `{Path(lf['path']).name}` ({lf['size_mb']} MB)\n")

            f.write("\n")

            for subdir in stats.get("subdirectories", [])[:20]:  # Limit subdirectories
                write_directory(subdir, indent + 1)

        for directory in scan_results.get("directories", [])[:50]:  # Limit top-level
            write_directory(directory)

    print(f"Saved Markdown report to: {md_path}")
    print("\n" + "=" * 60)
    print("Directory scan complete!")
    print(
        f"Summary: {summary['total_directories_scanned']} directories, "
        f"{summary['total_files']:,} files, {summary['total_size_gb']:.2f} GB"
    )


if __name__ == "__main__":
    main()
