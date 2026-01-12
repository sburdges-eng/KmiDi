# KmiDi Quick Start Guide

**Get making music in 5 minutes!**

---

## Prerequisites Checklist

Before starting, ensure you have:
- [ ] Python 3.9+ installed (`python --version`)
- [ ] pip installed and up to date
- [ ] Node 18+ installed (for desktop/web frontend, optional)
- [ ] Git installed (if cloning repository)

---

## Installation (2 minutes)

### Step 1: Install Python Package

```bash
# Navigate to project root
cd /path/to/KmiDi-1/KMiDi_PROJECT

# Install with development dependencies
pip install -e ".[dev]"

# Or install with all optional dependencies
pip install -e ".[all]"
```

### Step 2: Verify Installation

```bash
# Check Python version (should be 3.9+)
python --version

# Test imports
python -c "import music_brain; print('✓ music_brain installed')"
python -c "from music_brain.structure.comprehensive_engine import TherapySession; print('✓ core modules available')"
```

---

## First Run (3 minutes)

### Option A: API Server (Recommended for First Time)

```bash
# Terminal 1: Start API server
cd api
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000

# Terminal 2: Test the API
curl http://127.0.0.1:8000/health
```

**Expected Output:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": 1234567890.123,
  "services": {
    "music_brain": {"available": true, "version": "1.0.0"},
    "api": true
  },
  "system": {
    "cpu_percent": 15.5,
    "memory_percent": 60.2,
    "memory_available_mb": 4096.0
  }
}
```

### Option B: Python CLI (Direct)

```bash
# Generate music directly from command line
python -c "
from music_brain.structure.comprehensive_engine import TherapySession

session = TherapySession()
affect = session.process_core_input('I feel peaceful and calm')
session.set_scales(motivation=7, chaos=0.5)
plan = session.generate_plan()
print(f'Generated: {plan.chord_symbols} in {plan.mode} at {plan.tempo_bpm} BPM')
"
```

---

## Essential Commands Reference

### API Server

```bash
# Start server (development)
cd api && python -m uvicorn main:app --reload

# Start server (production)
cd api && uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# Check health
curl http://127.0.0.1:8000/health

# List emotions
curl http://127.0.0.1:8000/emotions
```

### Music Brain CLI (`daiw`)

```bash
# Extract groove from MIDI
daiw extract drums.mid

# Analyze chord progression
daiw analyze --chords song.mid

# Diagnose harmonic issues
daiw diagnose "F-C-Am-Dm"

# Create intent template
daiw intent new --title "My Song"

# Suggest rule-breaking options
daiw intent suggest grief

# Interactive teaching mode
daiw teach rulebreaking

# Generate music from a specific emotion
daiw generate "I feel peaceful"
```

### Testing

```bash
# Run Python tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=music_brain --cov-report=term-missing

# Run specific test file
pytest tests/test_intent_processor.py -v

# Run C++ tests (requires build)
cd build && ctest --output-on-failure
```

### Building C++ Components (Optional)

```bash
# Configure build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -G Ninja

# Build penta_core library
ninja penta_core

# Build and run tests
ninja penta_tests
./penta_tests
```

---

## Common Workflows

### Workflow 1: Generate Music from Emotion

```bash
# 1. Start API server
cd KMiDi_PROJECT/api && python -m uvicorn main:app --reload

# 2. Generate music
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "intent": {
      "emotional_intent": "I feel peaceful and calm, like a quiet morning"
    },
    "output_format": "midi"
  }'
```

### Workflow 2: Analyze Existing MIDI

```bash
# Analyze chord progression
daiw analyze --chords my_song.mid

# Extract groove
daiw extract my_song.mid --output groove.json

# Apply genre template
daiw apply --genre funk my_song.mid --output funk_version.mid
```

### Workflow 3: Intent-Driven Composition

```bash
# 1. Create intent template
daiw intent new --title "Grief Song"

# 2. Fill in Phase 0 (core wound/desire)
# Edit the generated JSON file

# 3. Get suggestions for rule-breaking
daiw intent suggest grief

# 4. Generate music using the intent
python -m music_brain.cli generate --intent intent.json
```

### Workflow 4: Development Cycle

```bash
# 1. Make code changes
vim music_brain/structure/generator.py

# 2. Run tests
pytest tests/ -v

# 3. Format code
black music_brain/

# 4. Type check
mypy music_brain/

# 5. Test changes
python -m music_brain.cli generate --emotion "sad"
```

---

## Troubleshooting Quick Tips

### Issue: "ModuleNotFoundError: No module named 'music_brain'"

**Solution:**
```bash
# Ensure you're in project root and package is installed
pip install -e .
python -c "import music_brain"
```

### Issue: "API server not starting"

**Solution:**
```bash
# Check if port 8000 is in use
lsof -i :8000

# Try different port
uvicorn main:app --port 8001

# Check for missing dependencies
pip install -e ".[all]"
```

### Issue: "Music Brain service unavailable" (503 error)

**Solution:**
```bash
# Verify Music Brain imports work
python -c "from music_brain.structure.comprehensive_engine import TherapySession"

# Check if all data files are present
ls data/emotions/
ls data/progressions/

# Reinstall if needed
pip install -e . --force-reinstall
```

### Issue: "C++ build failures"

**Solution:**
```bash
# Check CMake version (requires 3.27+)
cmake --version

# Check compiler
g++ --version  # or clang++

# Clean and rebuild
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

### Issue: "Import errors with optional dependencies"

**Solution:**
```bash
# Install all optional dependencies
pip install -e ".[all]"

# Or install specific groups
pip install -e ".[audio]"    # Audio processing
pip install -e ".[dev]"      # Development tools
pip install -e ".[mcp]"      # MCP servers
```

---

## Quick Configuration

### Environment Variables

Create a `.env` file in project root (or copy from `env.example`):

```bash
# Audio data directory
KELLY_AUDIO_DATA_ROOT=/path/to/audio/data

# API configuration
API_HOST=127.0.0.1
API_PORT=8000
LOG_LEVEL=INFO

# CORS (for web frontend)
CORS_ORIGINS=http://localhost:3000,http://localhost:1420
```

### Build Configuration

For different hardware setups, see `config/` directory:
- `build-dev-mac.yaml` - Apple Silicon (M-series) development
- `build-train-nvidia.yaml` - NVIDIA GPU training
- `build-prod-aws.yaml` - AWS production deployment

---

## Next Steps

1.  **Explore API**: Visit `http://127.0.0.1:8000/docs` when API is running to explore the interactive API documentation.
2.  **Try CLI Examples**: Experiment with the `daiw` CLI commands as outlined in this guide.
3.  **Launch Desktop App**: Run the KMiDi desktop application and explore its GUI.
4.  **Read Main Documentation**: See `PROJECT_MAIN.md` for a comprehensive overview of the project.
5.  **Join Development**: Refer to `CLAUDE.md` for a detailed AI assistant collaboration guide.

---

## Getting Help

- **Documentation**: See `PROJECT_MAIN.md` for a comprehensive overview and `PROJECT_BACKUP_CORE_CRITERIA.md` for core criteria.
- **API Docs**: Access interactive API documentation at `http://127.0.0.1:8000/docs` (when the server is running).
- **Issues**: Check existing issues or create a new one on the project repository.
- **Development Guide**: Refer to `CLAUDE.md` for a detailed guide for AI assistants.

---

## Checklist: Your First Song

- [x] Installed Python package (`pip install -e ".[all]"`)
- [x] Started API server (`uvicorn main:app --reload`)
- [x] Tested health endpoint (`curl http://127.0.0.1:8000/health`)
- [ ] Generated first music (`curl -X POST /generate ...`)
- [ ] Explored emotion list (`curl http://127.0.0.1:8000/emotions`)
- [ ] Read main documentation (`PROJECT_MAIN.md`)

**Congratulations!** You're ready to create music with KmiDi! 🎵
