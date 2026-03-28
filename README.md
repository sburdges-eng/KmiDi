# KMiDi

Emotion-driven music generation platform featuring 10 DAW plugins, 12 companion engines, and 23 ML model tasks. Built with React, Tauri, JUCE, Python, Rust, and C++.

## Architecture

- **music_brain/** -- Python ML core: emotion analysis, groove generation, audio processing
- **apps/kmidi/** -- Tauri desktop app with React frontend
- **libs/ai-core/** -- Shared ML and inference libraries
- **src/** -- JUCE C++ audio plugins and Rust bindings
- **src-tauri/** -- Rust Tauri host bindings
- **plugin/** -- JUCE audio/MIDI plugin implementation
- **docs/** -- Full documentation and ADRs

## Quickstart

### Prerequisites

- Node.js >= 18
- Python >= 3.9
- Rust toolchain (for Tauri builds)

### Install

```bash
# Frontend and Tauri dependencies
npm install

# Python environment (editable install)
pip install -e ".[dev,audio]"

# Run the dev server
npm run dev
```

### Running Tests

```bash
pytest
npm test
```

## V1 Build Paths

- **Pipeline A (penta_core + PyInstaller + Tauri):** `./scripts/build_v1.sh`
- **Pipeline B (KellyFFI + Tauri):** `./scripts/build-full-stack.sh` -- see [docs/FULL_STACK_BUILD.md](docs/FULL_STACK_BUILD.md)

## Development

Pre-commit hooks are configured for linting, formatting, type checking, and secret scanning:

```bash
pip install pre-commit
pre-commit install
```

Dev setup script: `./scripts/dev-setup.sh` -- see [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for details.

## Documentation

See [docs/](docs/) for detailed guides on model training, plugin development, companion engine APIs, and architecture decision records.

## License

MIT
