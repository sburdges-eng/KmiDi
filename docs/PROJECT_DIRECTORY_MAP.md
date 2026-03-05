# Project Directory Map

**Last Updated:** 2026-01-21

## 🟡 ACTIVE DEVELOPMENT (Yellow/Gold)

These directories contain active, maintained source code:

```
🟡 ACTIVE DEVELOPMENT:
├── src/                    [🟡 ACTIVE] Main source code
│   ├── plugin/            [🟡 ACTIVE] VST3/CLAP plugins
│   ├── gui/                [🟡 ACTIVE] Desktop application
│   ├── bridge/             [🟡 ACTIVE] FFI bindings
│   ├── core/               [🟡 ACTIVE] Core library
│   └── stubs/              [🟡 ACTIVE] Test stubs
├── include/                [🟡 ACTIVE] Public headers
├── tests/                  [🟡 ACTIVE] Test suite
└── bindings/               [🟡 ACTIVE] Language bindings
```

### How to Identify Active Directories

Active directories contain `ACTIVE_DEVELOPMENT.md` files with 🟡 yellow/gold markers.

## 📦 ARCHIVED/REFERENCE

These directories are archived or for reference only:

```
📦 ARCHIVED/REFERENCE:
├── KmiDi_FINAL/            [📦 ARCHIVED] Original source location
│   └── engine/src/         [📦 ARCHIVED] Migrated from here
├── KmiDi_PROJECT/          [📦 ARCHIVED] Project reference
├── KmiDi_BACKUP/           [📦 ARCHIVED] Backup archive
└── _QUARANTINE_*/          [📦 ARCHIVED] Quarantined files
```

## Quick Navigation

### For Development
- **Plugin Development:** `src/plugin/`
- **GUI Development:** `src/gui/`
- **FFI/Bindings:** `src/bridge/`
- **Core Library:** `src/core/`

### For Reference
- **Original Sources:** `KmiDi_FINAL/engine/src/`
- **Project History:** `KmiDi_PROJECT/`

## Migration History

All active files in `src/` were migrated from `KmiDi_FINAL/engine/src/` on 2026-01-21.
See `PROJECT_SOURCE_MANIFEST.md` for complete details.
