# Canonical Folder Structure

This document defines the target folder schema for the KmiDi project. The structure is designed to keep source code, assets, configuration, and generated output clearly separated.

## Sacred Directories (Protected)

These directories contain essential project files and should be carefully maintained:

### `src/`
**Purpose**: All source code only

**Contents**:
- `python/` - Python source code
- `cpp/` - C++ source code  
- `juce/` - JUCE framework code
- `scripts/` - Executable scripts
- Other language-specific source subdirectories

**Rules**:
- Only source code files belong here
- No generated files, no build artifacts
- No data files, no configuration files

### `assets/`
**Purpose**: Non-generated, human-curated data

**Contents**:
- `audio/` - Audio files (WAV, MP3, etc.)
- `midi/` - MIDI files
- `images/` - Image files
- `datasets/` - Training datasets and sample data

**Rules**:
- Human-created or curated content only
- No generated or processed files
- No source code

### `models/`
**Purpose**: FINAL models only

**Contents**:
- `trained/` - Completed, trained models ready for use
- `imported/` - Models imported from external sources

**Rules**:
- Only production-ready models
- No checkpoints, no intermediate training artifacts
- No experimental or work-in-progress models

### `configs/`
**Purpose**: Configuration files only

**Contents**:
- `*.yaml` - YAML configuration files
- `*.json` - JSON configuration files
- `*.toml` - TOML configuration files

**Rules**:
- Configuration files only
- No source code, no data files
- Well-structured, documented configurations

### `docs/`
**Purpose**: Documentation

**Contents**:
- `*.md` - Markdown documentation
- `*.txt` - Plain text documentation
- Diagrams and reference materials

**Rules**:
- Documentation files only
- No source code, no generated reports
- Keep documentation up to date

### `tests/`
**Purpose**: Test files

**Contents**:
- Test scripts and test data
- Unit tests, integration tests
- Test fixtures

**Rules**:
- Test code and test data only
- No production source code
- Keep tests organized by feature/component

## Generated Directories (Disposable)

These directories contain generated or build output and can be safely deleted and regenerated:

### `build/`
**Purpose**: Build artifacts

**Rules**:
- Always gitignored
- Can be deleted and regenerated
- Contains compiled binaries, object files

### `cache/`
**Purpose**: Build and runtime caches

**Rules**:
- Always gitignored
- Safe to delete
- Contains temporary cached data

### `logs/`
**Purpose**: Log files

**Rules**:
- Always gitignored
- Can be deleted
- Contains application and build logs

### `checkpoints/`
**Purpose**: Training checkpoints (if applicable)

**Rules**:
- Always gitignored
- Intermediate training state
- Can be regenerated from source

### `node_modules/`
**Purpose**: Node.js dependencies

**Rules**:
- Always gitignored
- Can be regenerated with `npm install`
- No source code should be here

## Utility Directories

### `tools/`
**Purpose**: One-off utilities and converters

**Rules**:
- Standalone utility scripts
- Not part of main application
- One-off conversion or processing tools

### `external/`
**Purpose**: External dependencies and third-party code

**Rules**:
- Third-party libraries
- External code dependencies
- Keep separate from main source

## Rules and Guidelines

### General Principles

1. **Only `src/`, `assets/`, `models/`, `configs/`, and `docs/` are sacred**
   - These directories contain irreplaceable project files
   - Everything else is disposable or can be regenerated

2. **Everything else is disposable**
   - If a file doesn't clearly belong in a sacred directory, it's likely disposable
   - Generated output lives outside logic directories
   - Build artifacts should never be in source directories

3. **If a file doesn't clearly belong, it doesn't belong at all**
   - Unclear files should be reviewed or removed
   - Ambiguity suggests the file may not be needed
   - When in doubt, quarantine and test

### File Organization Rules

- **No mixing of concerns**: Source code, data, configs, and generated files should be separate
- **Clear ownership**: Each file should have one clear purpose and location
- **No ambiguity**: If you can't immediately tell where a file belongs, reconsider its necessity
- **Generated = Disposable**: Anything that can be regenerated should be in a generated directory or gitignored

### Migration Guidelines

When reorganizing files:

1. **Identify file type**: Source code, asset, config, generated, or disposable
2. **Find appropriate sacred directory**: If it doesn't fit, reconsider if it's needed
3. **Move to generated directories**: If it's generated, move to appropriate build/cache/log directory
4. **Delete or quarantine**: If unclear, quarantine first, test, then decide

### Enforcement

- Regular audits should identify files that don't fit this structure
- Quarantine scripts can safely move ambiguous files for review
- Generated directories should always be gitignored
- Sacred directories should never contain generated files

## Example Structure

```
KmiDi/
├── src/                    # Sacred: Source code only
│   ├── python/
│   ├── cpp/
│   └── scripts/
├── assets/                 # Sacred: Human-curated data
│   ├── audio/
│   ├── midi/
│   └── images/
├── models/                 # Sacred: Final models only
│   ├── trained/
│   └── imported/
├── configs/                # Sacred: Configuration files
│   ├── *.yaml
│   └── *.json
├── docs/                   # Sacred: Documentation
│   └── *.md
├── tests/                  # Test files
│   └── test_*.py
├── tools/                  # Utility scripts
│   └── converters/
├── external/               # Third-party code
│   └── libraries/
├── build/                  # Generated: Build artifacts (gitignored)
├── cache/                  # Generated: Caches (gitignored)
├── logs/                   # Generated: Log files (gitignored)
└── checkpoints/            # Generated: Training checkpoints (gitignored)
```

## Maintenance

This structure should be maintained through:
- Regular audits using `one_shot_audit.sh`
- Review of deletion plans before cleanup
- Quarantine system for safe file migration
- Clear documentation of any exceptions
