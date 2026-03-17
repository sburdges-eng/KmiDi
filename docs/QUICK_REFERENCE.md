# Quick Reference Guide

**Last Updated:** 2026-01-23

## 🚀 Quick Start

### 1. Verify Everything Works
```bash
# Check imports
python3 scripts/verify_imports.py

# Run integration tests
python3 scripts/test_python_integration.py

# Check build prerequisites
python3 scripts/verify_build.py
```

### 2. Set Up Build
```bash
# Canonical dev setup (recommended)
./scripts/dev-setup.sh

# Optional: ./scripts/setup_build.sh or see BUILD.md for manual CMake setup
```

### 3. Run Tests
```bash
# Python integration tests
python3 scripts/test_python_integration.py

# Import verification
python3 scripts/verify_imports.py
```

## 📁 Key Files

### Scripts
- `scripts/verify_imports.py` - Verify Python imports (15/15 ✅)
- `scripts/test_python_integration.py` - Integration tests (8/8 ✅)
- `scripts/verify_build.py` - Build prerequisites (7/7 ✅)
- `scripts/setup_build.sh` - Automated build setup

### Documentation
- `docs/START_HERE.md` - Quick start guide
- `docs/NEXT_DEVELOPMENT_PHASE.md` - Development roadmap
- `docs/WORKSPACE_SETUP.md` - Development environment
- `QUICK_START.md` - Repo quick start + Kelly Companion usage
- *(Legacy: BUILD_STATUS.md, PHASE_1_PROGRESS.md — not present in repo; see BUILD.md and docs/DEVELOPMENT.md for current build status.)*

## ✅ Current Status

### Python Modules
- ✅ All imports working (15/15)
- ✅ All integration tests passing (8/8)
- ✅ All modules verified

### Build System
- ✅ Prerequisites verified (7/7)
- ⚠️ JUCE setup needed
- ⚠️ CMake configuration pending

## 🔧 Common Commands

### Git
```bash
# Check status
git status

# View recent commits
git log --oneline -10

# Push to remote
git push origin 2026-01-10-k5of-53bf1
```

### Python
```bash
# Test imports
python3 -c "from music_brain.session.intent_schema import CompleteSongIntent; print('✅')"

# Test engines
python3 -c "from music_brain.kelly_companion.engines import BassEngine; print('✅')"
```

### Build
```bash
# Verify prerequisites
python3 scripts/verify_build.py

# Setup build
./scripts/setup_build.sh

# Build specific target
cmake --build build --target penta_core
```

## 📊 Test Results

### Import Tests: 15/15 ✅
- music_brain package
- Session modules
- Kelly Companion
- Engines
- Harmony system
- Groove system
- Orchestrator
- Intelligence
- Learning

### Integration Tests: 8/8 ✅
- Emotion Thesaurus
- Intent Processing
- Engine Imports
- Data Files
- Session Management
- Harmony System
- Groove System
- Orchestrator

### Build Checks: 7/7 ✅
- CMake
- Python headers
- pybind11
- Build directory
- Penta-core sources
- Python bindings
- Include headers

## 🐛 Troubleshooting

### Import Errors
```bash
# Run verification
python3 scripts/verify_imports.py

# Check Python path
python3 -c "import sys; print('\n'.join(sys.path))"
```

### Build Errors
```bash
# Check prerequisites
python3 scripts/verify_build.py

# See BUILD.md and docs/DEVELOPMENT.md for solutions
```

### Module Not Found
```bash
# Ensure you're in repo root (KmiDi)
cd /path/to/KmiDi

# Check if module exists
ls -la music_brain/
```

## 📝 Next Steps

1. **Set up build dependencies:**
   ```bash
   ./scripts/setup_build.sh
   ```

2. **Continue Phase 1.2:**
   - See `NEXT_DEVELOPMENT_PHASE.md`
   - Follow Phase 1.2 tasks

3. **Run end-to-end tests:**
   - See Phase 1.3 in roadmap

## 🔗 Useful Links

- **Development Roadmap:** `docs/NEXT_DEVELOPMENT_PHASE.md`
- **Build / dev:** `BUILD.md`, `docs/DEVELOPMENT.md`
- **Quick Start:** `QUICK_START.md`, `docs/START_HERE.md`

---

**Need Help?** Check the documentation files listed above or run the verification scripts.
