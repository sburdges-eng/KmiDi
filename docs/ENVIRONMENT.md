# Environment Configuration Guide

Complete reference for KmiDi development environment variables and configuration.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [File Structure](#file-structure)
- [Variable Categories](#variable-categories)
- [Setup Instructions](#setup-instructions)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)

## Overview

KmiDi uses a hybrid environment structure that combines:
- **Base configuration** (`.env`, `.env.development`, `.env.production`)
- **Feature-specific configs** (`env/.env.*`)
- **User overrides** (`.env.local` - git-ignored)

This structure provides:
- ✅ Isolation between features
- ✅ Security (secrets in git-ignored files)
- ✅ Flexibility (override any variable)
- ✅ Documentation (all variables in one place)
- ✅ Validation (automated checks)

## Quick Start

### 1. Initial Setup

```bash
# Run interactive setup wizard
./scripts/setup-env.sh

# Or manually copy template
cp .env.example .env
# Edit .env with your values
```

### 2. Load Environment

```bash
# Source the environment loader
source scripts/load-env.sh

# Or load specific features
source scripts/load-env.sh tauri ml
```

### 3. Validate Configuration

```bash
./scripts/validate-env.sh
```

## File Structure

```
KmiDi-1/
├── .env                    # Base environment (git-ignored)
├── .env.example            # Template (committed)
├── .env.local              # User overrides (git-ignored)
├── .env.development        # Development defaults (committed)
├── .env.production         # Production template (committed)
├── env/
│   ├── .env.tauri.example  # Tauri/Frontend config
│   ├── .env.ml.example     # ML/Python services
│   ├── .env.training.example # Training pipeline
│   ├── .env.mcp.example    # MCP server credentials
│   └── .env.build.example  # Build config reference
└── scripts/
    ├── load-env.sh         # Environment loader
    ├── setup-env.sh        # Interactive setup
    └── validate-env.sh     # Validation script
```

### Priority Order

Variables are loaded in this order (later files override earlier ones):

1. `.env` (base configuration)
2. Feature-specific files (`env/.env.tauri`, etc.)
3. `.env.local` (highest priority, user overrides)

## Variable Categories

### 1. API Keys & Secrets

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `OPENAI_API_KEY` | No | OpenAI API key | `sk-...` |
| `ANTHROPIC_API_KEY` | No | Anthropic API key | `sk-ant-...` |
| `GOOGLE_API_KEY` | No | Google API key | `AIza...` |
| `XAI_API_KEY` | No | xAI API key | `xai-...` |
| `GITHUB_TOKEN` | No | GitHub personal access token | `ghp_...` |

**Security**: These should be set in `.env.local` (git-ignored) or via your shell environment.

### 2. Paths & Directories

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `KELLY_MODELS_PATH` | Yes | `./models` | C++ model file location |
| `PYTHON_MODEL_PATH` | No | `./models` | Python model location |
| `TRAINING_DATA_PATH` | No | `./data/training` | Training dataset path |
| `CHECKPOINT_PATH` | No | `./checkpoints` | Model checkpoint directory |
| `LOG_PATH` | No | `./logs` | Log file directory |

**Note**: Paths can be absolute or relative to project root.

### 3. Service URLs & Ports

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TAURI_DEV_HOST` | No | `localhost` | Tauri dev server host |
| `TAURI_PLATFORM` | No | `macos` | Target platform (macos/windows/linux) |
| `KMIDI_API_URL` | No | `http://127.0.0.1:8000` | Backend API URL |
| `ML_INFERENCE_URL` | No | `http://127.0.0.1:8001` | ML inference service |
| `MCP_SERVER_PORT` | No | `3000` | MCP server port |

### 4. Feature Flags

| Variable | Default | Description |
|----------|---------|-------------|
| `KMIDI_USE_API` | `false` | Use FastAPI service (Streamlit) |
| `ENABLE_ML_INFERENCE` | `true` | Enable ML inference features |
| `ENABLE_MCP_SERVERS` | `true` | Enable MCP server features |

### 5. Debugging & Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `RUST_LOG` | `info` | Rust log level (error/warn/info/debug/trace) |
| `RUST_BACKTRACE` | `0` | Rust backtrace (0=off, 1=on) |
| `CXX_LOG_LEVEL` | `INFO` | C++ log level |
| `PYTHON_LOG_LEVEL` | `INFO` | Python log level |

**Development**: Set `RUST_LOG=debug` and `RUST_BACKTRACE=1` for detailed debugging.

### 6. Training Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LOCAL_RANK` | `0` | Local rank for distributed training |
| `WORLD_SIZE` | `1` | Total number of processes |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device IDs |
| `TRAINING_BATCH_SIZE` | `32` | Training batch size |

## Setup Instructions

### For New Developers

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd KmiDi-1
   ```

2. **Run setup wizard**
   ```bash
   ./scripts/setup-env.sh
   ```
   This will prompt for API keys and paths, creating `.env.local`.

3. **Validate setup**
   ```bash
   ./scripts/validate-env.sh
   ```

4. **Load environment** (in your shell)
   ```bash
   source scripts/load-env.sh
   ```

### For Python Development

Python code should use `python-dotenv` to load environment variables:

```python
from dotenv import load_dotenv
import os

# Load .env files
load_dotenv()  # Loads .env
load_dotenv('.env.local', override=True)  # Override with local

# Access variables
api_key = os.getenv('OPENAI_API_KEY')
models_path = os.getenv('KELLY_MODELS_PATH')
```

### For Rust/Tauri Development

Rust code can use the `dotenv` crate:

```rust
use dotenv::dotenv;
use std::env;

fn main() {
    dotenv().ok();  // Load .env
    
    let api_key = env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY not set");
}
```

Or load in `src-tauri/src/main.rs`:

```rust
#[cfg(not(target_os = "android"))]
fn main() {
    tauri::Builder::default()
        .setup(|app| {
            // Load environment variables
            dotenv::dotenv().ok();
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

### For C++ Development

C++ code reads environment variables directly:

```cpp
#include <cstdlib>

const char* models_path = std::getenv("KELLY_MODELS_PATH");
if (!models_path) {
    // Use default or error
}
```

### For Frontend Development

Vite automatically loads variables prefixed with `VITE_`:

```typescript
// vite.config.ts
const apiUrl = import.meta.env.VITE_API_URL;

// In components
const apiUrl = import.meta.env.VITE_API_URL;
```

## Usage Examples

### Loading Environment in Scripts

```bash
#!/bin/bash
# Load environment before running script
source scripts/load-env.sh

# Now variables are available
echo "Models path: $KELLY_MODELS_PATH"
python train.py
```

### Feature-Specific Loading

```bash
# Load only Tauri and ML features
source scripts/load-env.sh tauri ml

# Load all features (default)
source scripts/load-env.sh
```

### Development vs Production

```bash
# Development (verbose logging)
export RUST_LOG=debug
export RUST_BACKTRACE=1
source scripts/load-env.sh

# Production (minimal logging)
export RUST_LOG=warn
source scripts/load-env.sh
```

## Troubleshooting

### Variables Not Loading

**Problem**: Environment variables not available after sourcing.

**Solution**:
1. Check file exists: `ls -la .env .env.local`
2. Verify syntax: `scripts/validate-env.sh`
3. Ensure you're sourcing (not executing): `source scripts/load-env.sh`

### API Keys Not Working

**Problem**: API calls failing with authentication errors.

**Solution**:
1. Verify keys are set: `echo $OPENAI_API_KEY`
2. Check for placeholder values: `scripts/validate-env.sh`
3. Ensure keys are in `.env.local` (not committed to git)

### Path Not Found

**Problem**: `KELLY_MODELS_PATH` directory doesn't exist.

**Solution**:
1. Create directory: `mkdir -p models`
2. Or update path in `.env.local`:
   ```bash
   KELLY_MODELS_PATH=/path/to/your/models
   ```

### Conflicting Variables

**Problem**: Variable has unexpected value.

**Solution**: Check priority order. `.env.local` overrides everything. Use:
```bash
# Check what's set
env | grep VARIABLE_NAME

# See loading order
source scripts/load-env.sh
```

### Build Configuration

**Problem**: CMake options not working.

**Solution**: Build options are NOT environment variables. Set via CMake:
```bash
cmake -DBUILD_DESKTOP=ON -DBUILD_PLUGINS=ON ..
```

See `env/.env.build.example` for reference.

## Integration Points

### Python Services

Update Python entry points to load environment:

```python
# At the top of main scripts
from dotenv import load_dotenv
load_dotenv()
load_dotenv('.env.local', override=True)
```

### Rust/Tauri

Add to `Cargo.toml`:
```toml
[dependencies]
dotenv = "0.15"
```

Load in `main.rs` (see examples above).

### CMake

Document build options in `env/.env.build.example` but set via command line.

### Frontend

Use `VITE_` prefix for variables exposed to frontend:
```bash
VITE_API_URL=http://127.0.0.1:8000
```

## Best Practices

1. **Never commit secrets**: Use `.env.local` for API keys
2. **Use examples**: Keep `.env.example` up to date
3. **Validate regularly**: Run `validate-env.sh` before important tasks
4. **Document changes**: Update this file when adding new variables
5. **Test locally**: Verify changes work in `.env.local` before committing templates

## Additional Resources

- [Python dotenv documentation](https://pypi.org/project/python-dotenv/)
- [Rust dotenv crate](https://docs.rs/dotenv/)
- [Tauri environment variables](https://tauri.app/v1/guides/development/development-cycle)
- [Vite environment variables](https://vitejs.dev/guide/env-and-mode.html)
