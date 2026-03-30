#!/usr/bin/env python3
"""
KmiDi Project File Consolidation Script

Consolidates all project files into two main folders:
1. KmiDi_CODE_AND_CONFIG/ - Source code, configs, scripts, tests
2. KmiDi_DOCS_AND_DATA/ - Documentation, data files, guides

This script creates a consolidated structure while preserving file relationships.
"""

import os
import shutil
import json
from pathlib import Path
from typing import Dict, List, Set
import hashlib

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
CONSOLIDATED_ROOT = PROJECT_ROOT / "KmiDi_CONSOLIDATED"

# Directory structure
CODE_DIR = CONSOLIDATED_ROOT / "KmiDi_CODE_AND_CONFIG"
DOCS_DIR = CONSOLIDATED_ROOT / "KmiDi_DOCS_AND_DATA"

# File categories
CODE_EXTENSIONS = {
    '.py', '.cpp', '.h', '.hpp', '.c', '.cc', '.cxx',
    '.rs', '.ts', '.tsx', '.js', '.jsx', '.java', '.kt'
}

CONFIG_EXTENSIONS = {
    '.toml', '.yaml', '.yml', '.json', '.ini', '.cfg',
    '.cmake', '.cmake.in', '.mk', '.make', '.conf',
    '.env', '.env.example', '.properties', '.plist'
}

BUILD_EXTENSIONS = {
    '.cmake', 'CMakeLists.txt', '.ninja', '.gradle',
    'Makefile', 'makefile', '.mk'
}

SCRIPT_FILES = {
    '*.sh', '*.bash', '*.ps1', '*.bat', '*.cmd'
}

# Exclude patterns (audio training data, build artifacts, etc.)
EXCLUDE_PATTERNS = {
    # Audio files
    '.wav', '.mp3', '.flac', '.aiff', '.ogg', '.m4a', '.wma',
    # Training datasets (raw audio)
    'maestro', 'cremad', 'ravdess', 'tess', 'gtzan',
    # Build artifacts
    '__pycache__', '.pyc', '.pyo', '.pyd',
    'build/', 'dist/', '.egg-info/',
    # Model binaries
    '.bin', '.pth', '.ckpt', '.mlmodel', '.onnx', '.h5', '.pb',
    # Generated outputs
    'output/', 'checkpoints/',
    # Dependencies
    'node_modules/', 'venv/', '.venv/', 'env/',
    # Version control
    '.git/', '.gitignore',
    # IDE
    '.vscode/', '.idea/', '.cursor/',
    # OS
    '.DS_Store', 'Thumbs.db',
}

# Data files to include (schemas, not raw audio)
DATA_INCLUDE_PATTERNS = {
    '*.json',  # JSON schemas and configuration data
    '*.yaml',  # YAML configuration
    '*.yml',   # YAML configuration
}

def should_exclude(path: Path) -> bool:
    """Check if path should be excluded."""
    path_str = str(path).lower()

    # Check exclude patterns
    for pattern in EXCLUDE_PATTERNS:
        if pattern.lower() in path_str:
            # Exception: Don't exclude JSON/YAML config files even if in excluded dirs
            if path.suffix in {'.json', '.yaml', '.yml'} and 'config' in path_str:
                continue
            return True

    # Don't exclude our consolidated directories
    if 'KmiDi_CONSOLIDATED' in path_str:
        return False

    return False

def is_code_file(path: Path) -> bool:
    """Check if file is source code."""
    return path.suffix in CODE_EXTENSIONS or path.name in {'CMakeLists.txt', 'Makefile'}

def is_config_file(path: Path) -> bool:
    """Check if file is configuration."""
    if path.suffix in CONFIG_EXTENSIONS:
        # Exclude data JSON files that are large audio datasets
        if path.suffix == '.json':
            # Include JSON schemas and configs, exclude large data files
            path_str = str(path).lower()
            if any(exclude in path_str for exclude in ['maestro', 'cremad', 'ravdess', 'tess']):
                return False
        return True
    return False

def is_doc_file(path: Path) -> bool:
    """Check if file is documentation."""
    doc_extensions = {'.md', '.txt', '.rst', '.html', '.pdf', '.tex'}
    return path.suffix in doc_extensions

def is_data_file(path: Path) -> bool:
    """Check if file is data (schema/config, not raw audio)."""
    # JSON/YAML data schemas
    if path.suffix in {'.json', '.yaml', '.yml'}:
        path_str = str(path).lower()
        # Include schema and config files
        if any(keyword in path_str for keyword in ['schema', 'config', 'metadata', 'chord', 'scale', 'groove', 'progression', 'emotion', 'rule']):
            return True
    return False

def is_test_file(path: Path) -> bool:
    """Check if file is a test."""
    test_keywords = {'test_', '_test', 'spec', 'test.py', 'test.cpp', 'test.h'}
    return any(keyword in path.name.lower() for keyword in test_keywords) or 'tests' in path.parts

def is_script_file(path: Path) -> bool:
    """Check if file is a script."""
    if path.suffix in {'.sh', '.bash', '.ps1', '.bat', '.cmd', '.py'}:
        # Python files in scripts/ directory
        if 'scripts' in path.parts and path.suffix == '.py':
            return True
        # Shell scripts
        if path.suffix in {'.sh', '.bash'}:
            return True
    return False

def get_file_category(path: Path) -> str:
    """Determine file category for organization."""
    if should_exclude(path):
        return 'exclude'

    # Source code
    if is_code_file(path):
        return 'code'

    # Configuration
    if is_config_file(path):
        return 'config'

    # Scripts
    if is_script_file(path):
        return 'script'

    # Tests
    if is_test_file(path):
        return 'test'

    # Documentation
    if is_doc_file(path):
        return 'doc'

    # Data files (schemas)
    if is_data_file(path):
        return 'data'

    # Default: put in CODE if it's in a code directory, otherwise DOCS
    if any(keyword in path.parts for keyword in ['src', 'include', 'python', 'cpp', 'core']):
        return 'code'

    return 'doc'

def collect_files(root: Path) -> Dict[str, List[Path]]:
    """Collect all files and categorize them."""
    files = {
        'code': [],
        'config': [],
        'script': [],
        'test': [],
        'doc': [],
        'data': [],
        'exclude': [],
    }

    print(f"Scanning project root: {root}")

    for path in root.rglob('*'):
        if path.is_file():
            category = get_file_category(path)

            # Skip excluded files
            if category == 'exclude':
                continue

            # Skip files in consolidated directory
            if 'KmiDi_CONSOLIDATED' in str(path):
                continue

            files[category].append(path)

    return files

def create_consolidated_structure():
    """Create consolidated directory structure."""
    # Create root
    CONSOLIDATED_ROOT.mkdir(exist_ok=True)

    # Create main directories
    CODE_DIR.mkdir(exist_ok=True)
    DOCS_DIR.mkdir(exist_ok=True)

    # Code subdirectories
    (CODE_DIR / "source_code").mkdir(exist_ok=True)
    (CODE_DIR / "configurations").mkdir(exist_ok=True)
    (CODE_DIR / "scripts").mkdir(exist_ok=True)
    (CODE_DIR / "tests").mkdir(exist_ok=True)
    (CODE_DIR / "build_files").mkdir(exist_ok=True)

    # Docs subdirectories
    (DOCS_DIR / "documentation").mkdir(exist_ok=True)
    (DOCS_DIR / "data_schemas").mkdir(exist_ok=True)
    (DOCS_DIR / "guides").mkdir(exist_ok=True)

    print(f"Created consolidated structure at: {CONSOLIDATED_ROOT}")

def copy_file_preserve_structure(source: Path, dest_base: Path, category: str):
    """Copy file while preserving relative directory structure."""
    # Get relative path from project root
    try:
        rel_path = source.relative_to(PROJECT_ROOT)
    except ValueError:
        # File is outside project root, use full path
        rel_path = source.name

    # Determine destination based on category
    if category == 'code':
        dest = dest_base / "source_code" / rel_path
    elif category == 'config':
        dest = dest_base / "configurations" / rel_path
    elif category == 'script':
        dest = dest_base / "scripts" / rel_path
    elif category == 'test':
        dest = dest_base / "tests" / rel_path
    elif category == 'doc':
        dest = dest_base / "documentation" / rel_path
    elif category == 'data':
        dest = dest_base / "data_schemas" / rel_path
    else:
        dest = dest_base / "other" / rel_path

    # Create parent directories
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Copy file
    try:
        shutil.copy2(source, dest)
        return True, str(dest)
    except Exception as e:
        return False, str(e)

def create_manifest(files: Dict[str, List[Path]], manifest_path: Path):
    """Create manifest file listing all consolidated files."""
    manifest = {
        'consolidation_date': str(Path().cwd()),
        'total_files': sum(len(file_list) for file_list in files.values()),
        'categories': {},
        'structure': {
            'code_directory': str(CODE_DIR),
            'docs_directory': str(DOCS_DIR),
        }
    }

    for category, file_list in files.items():
        if category != 'exclude':
            manifest['categories'][category] = {
                'count': len(file_list),
                'files': [str(f.relative_to(PROJECT_ROOT)) for f in file_list[:100]]  # First 100
            }
            if len(file_list) > 100:
                manifest['categories'][category]['note'] = f'{len(file_list)} total files (showing first 100)'

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"Created manifest: {manifest_path}")

def main():
    """Main consolidation function."""
    print("=" * 80)
    print("KmiDi Project File Consolidation")
    print("=" * 80)
    print()

    # Check if consolidated directory already exists
    if CONSOLIDATED_ROOT.exists():
        response = input(f"Consolidated directory exists: {CONSOLIDATED_ROOT}\n"
                        f"Delete and recreate? (yes/no): ")
        if response.lower() == 'yes':
            shutil.rmtree(CONSOLIDATED_ROOT)
            print("Removed existing consolidated directory")
        else:
            print("Aborted. Exiting.")
            return

    # Create structure
    create_consolidated_structure()

    # Collect files
    print("\nCollecting files...")
    files = collect_files(PROJECT_ROOT)

    # Print summary
    print("\nFile Categories:")
    for category, file_list in files.items():
        if category != 'exclude':
            print(f"  {category:10}: {len(file_list):5} files")

    # Copy files
    print("\nCopying files to consolidated structure...")
    copy_results = {
        'success': 0,
        'failed': 0,
        'failed_files': []
    }

    for category, file_list in files.items():
        if category == 'exclude':
            continue

        print(f"\nCopying {category} files...")
        dest_base = CODE_DIR if category in ['code', 'config', 'script', 'test'] else DOCS_DIR

        for source_file in file_list:
            success, result = copy_file_preserve_structure(source_file, dest_base, category)
            if success:
                copy_results['success'] += 1
                if copy_results['success'] % 100 == 0:
                    print(f"  Copied {copy_results['success']} files...")
            else:
                copy_results['failed'] += 1
                copy_results['failed_files'].append((str(source_file), result))

    # Create manifest
    manifest_path = CONSOLIDATED_ROOT / "CONSOLIDATION_MANIFEST.json"
    create_manifest(files, manifest_path)

    # Create README
    readme_path = CONSOLIDATED_ROOT / "README.md"
    with open(readme_path, 'w') as f:
        f.write(f"""# KmiDi Consolidated Project Files

**Consolidated:** {Path().cwd()}
**Total Files:** {copy_results['success']}
**Failed:** {copy_results['failed']}

## Structure

- `KmiDi_CODE_AND_CONFIG/` - Source code, configurations, scripts, tests
  - `source_code/` - Python, C++, headers
  - `configurations/` - Config files (toml, yaml, json, cmake)
  - `scripts/` - Utility scripts
  - `tests/` - Test files
  - `build_files/` - Build configurations

- `KmiDi_DOCS_AND_DATA/` - Documentation and data
  - `documentation/` - Markdown, text docs
  - `data_schemas/` - JSON/YAML data schemas (not raw audio)
  - `guides/` - Production workflows, songwriting guides, etc.

## Excluded

- Audio training data (MAESTRO, CREMA-D, RAVDESS, TESS)
- Model binaries (.bin, .pth, .ckpt, .mlmodel, .onnx)
- Build artifacts (build/, dist/, __pycache__)
- Generated outputs (output/, checkpoints/)
- Dependencies (node_modules/, venv/)

## Usage

All project files are organized in the two main directories above.
Original relative paths are preserved within each category.

See `CONSOLIDATION_MANIFEST.json` for complete file listing.
""")

    # Summary
    print("\n" + "=" * 80)
    print("Consolidation Complete!")
    print("=" * 80)
    print(f"\nSuccessfully copied: {copy_results['success']} files")
    print(f"Failed: {copy_results['failed']} files")
    print(f"\nConsolidated structure: {CONSOLIDATED_ROOT}")
    print(f"\nManifest: {manifest_path}")
    print(f"README: {readme_path}")

    if copy_results['failed_files']:
        print("\nFailed files (first 10):")
        for file_path, error in copy_results['failed_files'][:10]:
            print(f"  {file_path}: {error}")

if __name__ == '__main__':
    main()
