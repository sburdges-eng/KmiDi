# Canonical Folder Structure

This document defines the target folder structure for the project. This structure prevents project rot by enforcing clear boundaries and making it obvious what belongs where.

## Folder Schema

```
PROJECT_ROOT/
├── src/                # All source code only
│   ├── python/
│   ├── cpp/
│   ├── juce/
│   └── scripts/
│
├── assets/             # Non-generated, human-curated data
│   ├── audio/
│   ├── midi/
│   ├── images/
│   └── datasets/
│
├── models/             # FINAL models only
│   ├── trained/
│   └── imported/
│
├── configs/            # yaml/json/toml only
│
├── docs/               # markdown, diagrams
│
├── tools/              # one-off utilities, converters
│
├── build/              # GENERATED (gitignored)
├── cache/              # GENERATED (gitignored)
├── logs/               # GENERATED (gitignored)
├── checkpoints/        # GENERATED (gitignored)
│
├── tests/
├── .gitignore
└── README.md
```

## Rules (Non-Negotiable)

### Sacred Directories
Only the following directories are considered sacred and should contain important, version-controlled files:

- `src/` - All source code only
- `assets/` - Non-generated, human-curated data
- `models/` - FINAL models only
- `configs/` - Configuration files (yaml/json/toml only)
- `docs/` - Documentation (markdown, diagrams)

### Disposable Directories
Everything else is disposable:

- `build/` - Build artifacts (gitignored)
- `cache/` - Cache files (gitignored)
- `logs/` - Log files (gitignored)
- `checkpoints/` - Model checkpoints (gitignored)
- Any other generated output

### Principles

1. **Generated output lives outside logic** - Build artifacts, caches, logs, and checkpoints must be in clearly marked directories that are gitignored.

2. **If a file doesn't clearly belong, it doesn't belong at all** - Ambiguity is a sign that something is wrong with the organization.

3. **Source code only in src/** - All executable code, scripts, and source files go under `src/` organized by language or framework.

4. **Assets are curated, not generated** - The `assets/` directory contains human-created or carefully selected data files, not automatically generated content.

5. **Models are final** - Only completed, production-ready models belong in `models/`. Training checkpoints and intermediate models belong in `checkpoints/` (gitignored).

## Migration Strategy

When migrating to this structure:

1. **One folder at a time** - Migrate deliberately, not recursively. Focus on one category at a time.

2. **Use the audit tools** - Run `one_shot_audit.sh` first to understand what you have.

3. **Quarantine, don't delete** - Use `quarantine_move.sh` to safely move files, then test before committing to deletion.

4. **Validate after each step** - Run builds and tests after each major migration to ensure nothing breaks.

## Enforcement

This structure is enforced through:

- Clear documentation (this file)
- `.gitignore` patterns for generated content
- Audit scripts to identify violations
- Quarantine tools to safely reorganize

Remember: **Not by motivation. By fences.** This structure prevents project rot by making it obvious when something doesn't belong.
