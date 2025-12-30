# KmiDi Architecture & Code Review
**Date**: 2025-12-30
**Reviewer**: Claude Code
**Scope**: Full codebase review + ML model analysis

---

## Executive Summary

✅ **Overall Grade: A-**

Your KmiDi project demonstrates excellent software engineering practices with a well-structured monorepo combining Python ML, C++ real-time audio, and multi-AI orchestration. Recent Mac development staging changes show mature DevOps thinking. Main issues are **organizational** (directory naming, config alignment) rather than code quality problems.

---

## 1. ML Models Analysis

### 1.1 Current Model Registry Status

**Location**: `models/registry.json`

All **7 models** are currently **STUBS** (no trained weights):

| Model | Status | Input→Output | Target | Params | Priority |
|-------|--------|--------------|--------|--------|----------|
| **EmotionRecognizer** | 🟡 Stub | 128→64 | <5ms | ~500K | 🔴 HIGH |
| **MelodyTransformer** | 🟡 Stub | 64→128 | <5ms | ~400K | 🔴 HIGH |
| **HarmonyPredictor** | 🟡 Stub | 128→64 | <3ms | ~100K | 🔴 HIGH |
| **DynamicsEngine** | 🟡 Stub | 32→16 | <1ms | ~20K | 🟢 MEDIUM |
| **GroovePredictor** | 🟡 Stub | 64→32 | <2ms | ~25K | 🟢 MEDIUM |
| **InstrumentRecognizer** | 🟡 Stub | 128→160 (dual-head) | <10ms | ~2M | 🟡 LOW |
| **EmotionNodeClassifier** | 🟡 Stub | 128→258 (6×6×6) | <15ms | ~3M | 🟡 LOW |

### 1.2 External SSD Search Results

**Finding**: No Extreme SSD currently mounted or accessible in the environment.

**Config References to External Storage**:
- `configs/emotion_recognizer.yaml:38` → `/Volumes/Extreme SSD/kelly-audio-data`
- `configs/harmony_predictor.yaml:33` → `/Volumes/Extreme SSD/kelly-audio-data`

**⚠️ Issue**: Configs reference external SSD that's not accessible in current dev container.

### 1.3 Existing Training Results

**Found**: `checkpoints/emotionrecognizer/results.json`

**Poor Performance** (1 epoch smoke test):
```json
{
  "test_results": {
    "accuracy": 0.1,          // 10% accuracy (worse than random for 7 classes!)
    "macro_f1": 0.026,        // Extremely low F1 score
    "best_val_loss": 1.949
  },
  "config": {
    "epochs": 1,              // Only 1 epoch trained
    "batch_size": 16,
    "device": "auto"
  }
}
```

**Analysis**: This is a genuine stub/smoke test, not a real trained model.

### 1.4 Available Data Assets

**Music Theory Data** (in `data/`):
- ✅ `chord_progressions_db.json` - Chord progression database
- ✅ `scales_database.json` - Scale definitions
- ✅ `scale_emotional_map.json` - Emotion-to-scale mapping
- ✅ `data/emotion_thesaurus/` - 6×6×6 emotion taxonomy
- ✅ `data/rules/rule_breaking_database.json` - Creative rule-breaking patterns
- ✅ `data/music_theory/` - Learning paths, exercises, concepts

**Missing**:
- ❌ Audio datasets for EmotionRecognizer
- ❌ MIDI datasets for MelodyTransformer
- ❌ Chord progression training data
- ❌ Groove/dynamics training data

---

## 2. Configuration Analysis

### 2.1 Two Config Purposes (Not a Bug!)

After deep review, I see **two distinct config types**:

#### **A. Build Configs** (`config/build-*.yaml`)
**Purpose**: Environment/infrastructure setup
**Examples**:
- `config/build-dev-mac.yaml` - Mac development environment
- `config/build-train-nvidia.yaml` (referenced but not present)
- `config/build-prod-aws.yaml` (referenced but not present)

**Content**:
```yaml
build: dev-mac
device: mps
python: 3.11
paths: ...
performance: ...
api: ...
```

#### **B. Training Configs** (`configs/<model>_*.yaml`)
**Purpose**: Model-specific training hyperparameters
**Examples**:
- `configs/emotion_recognizer.yaml` - Full training config
- `configs/melody_transformer.yaml` - Full training config
- `configs/train-mac-smoke.yaml` - Minimal smoke test config

**Content**:
```yaml
model_id: emotionrecognizer
architecture_type: cnn
epochs: 100
batch_size: 16
learning_rate: 0.001
...
```

### 2.2 Recommended Config Organization

**Option 1: Keep Separate (Recommended)**
```
config/                    # Build/environment configs
├── build-dev-mac.yaml
├── build-train-nvidia.yaml
└── build-prod-aws.yaml

configs/                   # Model training configs
├── emotion_recognizer.yaml
├── melody_transformer.yaml
├── train-mac-smoke.yaml
└── ...
```

**Pros**: Clear separation of concerns
**Cons**: Two directories (but for different purposes)

**Option 2: Consolidate with Subdirectories**
```
config/
├── builds/
│   ├── dev-mac.yaml
│   ├── train-nvidia.yaml
│   └── prod-aws.yaml
└── training/
    ├── emotion_recognizer.yaml
    ├── melody_transformer.yaml
    └── smoke/
        └── train-mac-smoke.yaml
```

**Pros**: Single top-level directory
**Cons**: More complex structure, requires path updates

### 2.3 Config Values Comparison

#### `config/build-dev-mac.yaml`
```yaml
training:
  enabled: false          # Inference only
  batch_size: 4          # Conservative
  fp16: true
  gradient_checkpointing: true
```

#### `configs/train-mac-smoke.yaml`
```yaml
model_id: emotionrecognizer
batch_size: 4            # Matches build config
epochs: 3                # Ultra-minimal for smoke
device: mps
notes: "Smoke test on M-series; clamps further if device guard triggers"
```

#### `scripts/train.py::enforce_device_constraints()`
```python
max_epochs = 5 if device.type == "mps" else min(config.epochs, 10)
max_batch = 8 if device.type == "mps" else 16
```

**Analysis**: Three-layer defense system:
1. **Build config**: Says training disabled by default (`enabled: false`)
2. **Smoke config**: Ultra-conservative (3 epochs, batch=4)
3. **Enforcement**: Safety net (clamps to max 5 epochs, batch=8 on MPS)

**Recommendation**: This is actually **good design**! Different use cases:
- Build config = "Should I allow training at all on this machine?"
- Smoke config = "Minimal smoke test values"
- Enforcement = "Hard limits to prevent OOM"

---

## 3. Code Quality Issues

### 3.1 CRITICAL: YAML Indentation (Tabs vs Spaces)

**File**: `config/build-dev-mac.yaml`
**Lines**: 11-42

**Problem**:
```yaml
paths:
	data_root: ${DATA_ROOT:-data}  # ← TAB character (bad!)
```

**Fix**:
```bash
sed -i 's/\t/  /g' config/build-dev-mac.yaml
```

### 3.2 Missing Type Hints

**File**: `scripts/train.py:211`

**Current**:
```python
def enforce_device_constraints(config: TrainConfig, device) -> None:
```

**Recommended**:
```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch

def enforce_device_constraints(config: TrainConfig, device: torch.device) -> None:
```

### 3.3 Magic Numbers

**File**: `scripts/train.py:215-216`

**Current**:
```python
max_epochs = 5 if device.type == "mps" else min(config.epochs, 10)
max_batch = 8 if device.type == "mps" else 16
```

**Recommended**:
```python
# At module level
MAX_EPOCHS_MPS_SMOKE = 5
MAX_EPOCHS_CPU_SMOKE = 10
MAX_BATCH_MPS = 8
MAX_BATCH_CPU = 16

# In function
max_epochs = MAX_EPOCHS_MPS_SMOKE if device.type == "mps" else min(config.epochs, MAX_EPOCHS_CPU_SMOKE)
max_batch = MAX_BATCH_MPS if device.type == "mps" else MAX_BATCH_CPU
```

---

## 4. Missing Model Configs

You have **14 configs** but only **7 models** in registry. Missing configs for:

### Should Create:
1. ✅ `configs/dynamics_engine.yaml` - EXISTS!
2. ✅ `configs/groove_predictor.yaml` - EXISTS!
3. ✅ `configs/instrument_recognizer.yaml` - EXISTS!
4. ✅ `configs/emotion_node_classifier.yaml` - EXISTS!

**Finding**: All model configs already exist! Good coverage.

### Additional Configs Found:
- `configs/music_foundation_base.yaml` - Foundation model (new?)
- `configs/macOS_16gb_optimized.yaml` - Hardware optimization
- `configs/nvidia_cuda_optimized.yaml` - Hardware optimization
- `configs/laptop_m4_small.yaml` - Hardware optimization

---

## 5. Recommended New Models

Based on your architecture and the "Interrogate Before Generate" philosophy, consider adding:

### Priority 1: Intent Classifier
**Purpose**: Map user text input → emotional intent
**Architecture**: BERT/DistilBERT fine-tune
**Input**: Text prompt ("I'm feeling lost and searching")
**Output**: Intent embedding (64-dim) → links to EmotionNodeClassifier

**Why**: Bridges natural language to music_brain intent schema

### Priority 2: Audio Feel Analyzer
**Purpose**: Extract "feel" from reference audio
**Architecture**: CNN + RNN for temporal analysis
**Input**: Raw audio waveform
**Output**: Feel descriptor (tempo, energy, density, texture)

**Why**: Enables "make it sound like this" workflows

### Priority 3: Structure Predictor
**Purpose**: Suggest song structure from intent
**Architecture**: Transformer for sequence generation
**Input**: Emotional arc + genre + duration
**Output**: Section sequence (intro→verse→chorus→bridge→outro)

**Why**: Helps users scaffold complete compositions

### Priority 4: Micro-Timing Model
**Purpose**: Humanize quantized MIDI
**Architecture**: Small RNN (timing deviations)
**Input**: Quantized note sequence + groove style
**Output**: Timing micro-adjustments (±ms)

**Why**: Adds expressive timing to generated MIDI

---

## 6. Data Strategy Recommendations

### Immediate (Week 1-2):
1. **Synthetic Data Generation**
   - Generate synthetic emotion audio using TTS + effects
   - Create MIDI datasets from chord_progressions_db.json
   - Use music21 to generate training data from theory rules

2. **Public Datasets**
   - **RAVDESS** (emotion audio) - 7 emotions, ~1500 clips
   - **Lakh MIDI** - 176K MIDI files for melody/harmony
   - **FMA** (Free Music Archive) - 106K tracks for groove extraction

3. **Minimal Viable Training**
   - Start with 1K samples per model
   - Focus on EmotionRecognizer + HarmonyPredictor (highest priority)
   - Use transfer learning where possible

### Medium Term (Month 1-3):
1. **User-Generated Data Collection**
   - Add opt-in telemetry to iDAW plugins
   - Collect anonymized usage patterns
   - Build feedback loop: user rates AI suggestions

2. **Data Augmentation Pipeline**
   - Audio: pitch shift, time stretch, noise injection
   - MIDI: transposition, rhythmic variation, chord substitution

3. **Active Learning**
   - Identify low-confidence predictions
   - Request user feedback on edge cases
   - Continuously retrain with new labels

---

## 7. Training Priority Matrix

| Model | Data Availability | Complexity | Impact | Priority |
|-------|-------------------|------------|--------|----------|
| **HarmonyPredictor** | 🟢 High (have progressions DB) | 🟢 Low (MLP) | 🔴 Critical | **START HERE** |
| **EmotionRecognizer** | 🟡 Medium (RAVDESS) | 🟡 Medium (CNN) | 🔴 Critical | **2nd** |
| **MelodyTransformer** | 🟢 High (Lakh MIDI) | 🟡 Medium (Transformer) | 🔴 Critical | **3rd** |
| **GroovePredictor** | 🟡 Medium (FMA) | 🟡 Medium (RNN) | 🟡 Important | **4th** |
| **DynamicsEngine** | 🔴 Low (needs custom) | 🟢 Low (small MLP) | 🟡 Important | **5th** |
| **InstrumentRecognizer** | 🟡 Medium (NSynth) | 🔴 High (dual-head CNN) | 🟢 Nice-to-have | **Later** |
| **EmotionNodeClassifier** | 🔴 Low (needs 6×6×6 data) | 🔴 High (multi-head) | 🟢 Nice-to-have | **Later** |

---

## 8. Architecture Strengths

✅ **Excellent Design Decisions**:
1. **Dual-Engine Architecture** (Python Side B / C++ Side A)
2. **Hardware-Aware Build System** (dev-mac, train-nvidia, prod-aws)
3. **Device Constraint Enforcement** (prevents OOM on Mac)
4. **Environment Variable Portability** (`${VAR:-default}`)
5. **Comprehensive Model Registry** (tracks training metadata)
6. **Fallback Heuristics** (system works even without trained models)
7. **Multi-AI Orchestration** (MCP workstation/todo/swarm)
8. **Intent-Driven Philosophy** ("Interrogate Before Generate")

---

## 9. Critical Issues (Fix Before Production)

🔴 **P0 - Critical**:
1. Fix YAML tabs→spaces in `config/build-dev-mac.yaml`
2. Verify external SSD path strategy (currently broken in dev container)
3. Add missing fields to smoke config (`num_workers`, `pin_memory`)

🟡 **P1 - High**:
4. Add type hints to `enforce_device_constraints()`
5. Extract magic numbers to named constants
6. Update `pyproject.toml` testpaths to include all test directories
7. Create at least ONE trained model (suggest: HarmonyPredictor)

🟢 **P2 - Medium**:
8. Decide on config directory strategy (keep separate or consolidate)
9. Update BUILD_VARIANTS.md to match new env-var approach
10. Add pre-commit hooks for Black/Flake8
11. Create .env.example with documented variables

---

## 10. Proposed Edits

### Edit 1: Fix YAML Indentation
**File**: `config/build-dev-mac.yaml`

**Issue**: Tabs instead of spaces (lines 11-42)

**Proposed Change**:
```yaml
# Replace all tabs with 2 spaces
paths:
  data_root: ${DATA_ROOT:-data}
  models_root: ${MODELS_ROOT:-models}
  output_root: ${OUTPUT_ROOT:-output}
  checkpoints: ${MODELS_ROOT:-models}/checkpoints

performance:
  target_latency_ms: 150
  profile_enabled: true
  memory_monitor: true
  max_batch_size: 16
# ... etc
```

**Justification**: YAML spec requires consistent spacing; tabs may break parsers.

---

### Edit 2: Add Missing Fields to Smoke Config
**File**: `configs/train-mac-smoke.yaml`

**Current** (22 lines):
```yaml
# ... existing content ...
device: mps
notes: "Smoke test on M-series; clamps further if device guard triggers"
```

**Proposed Addition**:
```yaml
# ... existing content ...
device: mps
num_workers: 0       # Required for Mac compatibility
pin_memory: false    # Not beneficial on MPS
notes: "Smoke test on M-series; clamps further if device guard triggers"
```

**Justification**: `enforce_device_constraints()` expects these fields; prevents AttributeError.

---

### Edit 3: Add Type Hints and Constants
**File**: `scripts/train.py`

**Current** (lines 211-243):
```python
def enforce_device_constraints(config: TrainConfig, device) -> None:
    """Clamp config for resource-limited devices (e.g., MPS/CPU smoke)."""

    if device.type in {"mps", "cpu"}:
        max_epochs = 5 if device.type == "mps" else min(config.epochs, 10)
        max_batch = 8 if device.type == "mps" else 16
```

**Proposed** (add after imports, update function):
```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch

# ... later in file, after ROOT definitions ...

# Device constraint limits (smoke testing)
MAX_EPOCHS_MPS_SMOKE = 5
MAX_EPOCHS_CPU_SMOKE = 10
MAX_BATCH_MPS = 8
MAX_BATCH_CPU = 16

# ... in function ...

def enforce_device_constraints(config: TrainConfig, device: "torch.device") -> None:
    """Clamp config for resource-limited devices (e.g., MPS/CPU smoke)."""

    if device.type in {"mps", "cpu"}:
        max_epochs = MAX_EPOCHS_MPS_SMOKE if device.type == "mps" else min(config.epochs, MAX_EPOCHS_CPU_SMOKE)
        max_batch = MAX_BATCH_MPS if device.type == "mps" else MAX_BATCH_CPU
```

**Justification**: Adds type safety, improves maintainability, makes limits configurable.

---

## 11. Model Training Quickstart

### Step 1: Train HarmonyPredictor (Easiest)

**Why First**: You already have data (`data/chord_progressions_db.json`)

```bash
# 1. Prepare dataset from existing JSON
python scripts/prepare_chord_dataset.py  # You may need to create this

# 2. Run training with full config
python scripts/train.py --config configs/harmony_predictor.yaml

# 3. Validate output
ls checkpoints/harmony_predictor/
# Should see: best_model.pt, best_model.onnx, training_log.json
```

### Step 2: Train EmotionRecognizer

```bash
# 1. Download RAVDESS dataset
python scripts/bulk_download_data.py --dataset ravdess --output data/emotion_audio

# 2. Run training
python scripts/train.py --config configs/emotion_recognizer.yaml

# 3. Export for C++
python scripts/export_to_rtneural.py \
  --checkpoint checkpoints/emotion_recognizer/best_model.pt \
  --output models/emotionrecognizer.json
```

### Step 3: Integrate into iDAW_Core

```cpp
// In C++ plugin
#include "penta/ml/MLInterface.h"

auto emotionModel = penta::ml::MLInterface::loadModel("emotionrecognizer.json");
auto embedding = emotionModel->predict(audioFeatures);  // 128→64
```

---

## 12. Final Recommendations

### DO IMMEDIATELY:
1. ✅ Fix YAML tabs (5 min fix)
2. ✅ Add `num_workers`/`pin_memory` to smoke config (1 min)
3. ✅ Create GitHub issues (done - see `.github/ISSUE_TEMPLATE/`)

### DO THIS WEEK:
4. Train HarmonyPredictor (you have the data!)
5. Add type hints + constants to train.py
6. Decide on config directory strategy
7. Test smoke config end-to-end

### DO THIS MONTH:
8. Download RAVDESS, train EmotionRecognizer
9. Download Lakh MIDI, train MelodyTransformer
10. Set up CI/CD for model training
11. Create baseline performance benchmarks

### CONSIDER FOR FUTURE:
12. Add Intent Classifier model
13. Build data collection telemetry
14. Implement active learning loop
15. Add 4 new models (see section 5)

---

## Conclusion

Your architecture is **solid**. The "Interrogate Before Generate" philosophy is well-implemented through the intent schema and rule-breaking system. Main gaps are:

1. **Trained models** (all currently stubs)
2. **Training datasets** (missing most)
3. **Minor config issues** (easy fixes)

**Next Critical Path**:
```
Fix YAML tabs → Train HarmonyPredictor → Train EmotionRecognizer → Integrate into C++
```

You're closer to production than it might feel - the infrastructure is excellent, you just need to populate it with trained weights.

---

**Questions for You**:
1. Do you want me to implement the 3 proposed edits above?
2. Should I create the `scripts/prepare_chord_dataset.py` helper?
3. Prefer to keep `config/` and `configs/` separate, or consolidate?
4. Which model should we train FIRST (I recommend HarmonyPredictor)?

Let me know how you'd like to proceed!
