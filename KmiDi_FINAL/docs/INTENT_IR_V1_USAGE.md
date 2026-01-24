# Intent IR v1 Usage Guide

## Overview

Intent IR v1 is the canonical representation of musical intent in KmiDi. It provides a single, versioned, serializable format that all intent producers (Python, UI, ML) emit and all intent consumers (C++ engines) consume.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Python    │────▶│  IntentFrame │────▶│   Engines   │
│   UI/ML     │     │   (IR v1)    │     │  (C++/Rust) │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   JSON Log   │
                    │  (Debugging) │
                    └──────────────┘
```

## Key Principles

1. **Bias-based**: IR contains only musical biases/tendencies, not concrete notes/chords
2. **Immutable**: Once created, IntentFrame should not be modified
3. **Versioned**: Breaking changes increment `ir_version`
4. **Serializable**: Can be converted to/from JSON for logging and debugging
5. **Language-agnostic**: Works across Python, C++, Rust, Swift

## Usage Examples

### Python: Creating IntentFrame

```python
from music_brain.session.intent_ir import IntentFrame, IntentSource
from music_brain.session.intent_ir_converter import convert_complete_song_intent_to_ir

# Convert from CompleteSongIntent
complete_intent = ...  # Your CompleteSongIntent object
frame = convert_complete_song_intent_to_ir(
    complete_intent,
    session_id=12345,
    source=IntentSource.ML_TEXT
)

# Serialize to JSON
json_str = frame.to_json()
with open("intent.json", "w") as f:
    f.write(json_str)
```

### C++: Consuming IntentFrame

```cpp
#include "common/IntentIRExtractor.h"
#include "engines/MelodyEngine.h"

// Get IntentFrame from Python/UI
IntentFrame frame = ...;

// Extract parameters for your engine
MelodyParams params = MelodyParams::fromIntentFrame(frame);

// Use with engine
MelodyEngine engine;
MelodyOutput output = engine.generateFromIntentFrame(frame, "C", 4, 120);
```

### C++: Creating IntentFrame from IntentPipeline

```cpp
#include "engine/IntentPipeline.h"

IntentPipeline pipeline;
Wound wound{"I feel lost", 0.8f, "user_input"};

// New canonical API
IntentFrame frame = pipeline.processToIntentFrame(wound, session_id);

// Engines consume directly
MelodyEngine melodyEngine;
auto melody = melodyEngine.generateFromIntentFrame(frame, "C", 4, 120);
```

### Rust: Validating IntentFrame

```rust
use intent_ir::*;

let mut frame = IntentFrame::default();
// ... populate frame ...

// Validate
match validate_intent_frame(&frame) {
    Ok(()) => println!("Valid!"),
    Err(e) => println!("Invalid: {:?}", e),
}

// Clamp values
clamp_intent_frame(&mut frame);
```

## Field Mappings

### EmotionState → Engine Parameters

- `valence`: Negative = sad/minor, Positive = happy/major
- `arousal`: Low = calm/sparse, High = excited/dense
- `dominance`: Low = submissive, High = dominant
- `confidence`: Affects how strongly engines apply the emotion

### MusicalIntent → Engine Parameters

- `tempo_bias`: -1.0 (slow) → +1.0 (fast)
- `rhythmic_density`: 0.0 (sparse) → 1.0 (dense)
- `groove_strength`: 0.0 (loose) → 1.0 (tight)
- `harmonic_tension`: 0.0 (consonant) → 1.0 (tense)
- `mode_preference`: -1 (minor), 0 (neutral), +1 (major)
- `melodic_activity`: 0.0 (minimal) → 1.0 (active)
- `contour_variance`: 0.0 (flat) → 1.0 (wide)
- `dynamic_range`: 0.0 (flat) → 1.0 (wide)
- `texture_density`: 0.0 (thin) → 1.0 (thick)

## Migration Guide

### For Engine Authors

1. Add `generateFromIntentFrame()` method to your engine
2. Use `IntentIRExtractor` helpers to extract relevant parameters
3. Map IR biases to your engine's concrete parameters
4. Keep existing methods for backward compatibility

### For Intent Producers

1. Use `convert_complete_song_intent_to_ir()` for Python
2. Use `IntentPipeline::processToIntentFrame()` for C++
3. Always validate/clamp before sending to engines
4. Set appropriate `IntentSource` for debugging

## Versioning

- Current version: **1**
- Version mismatches fail loudly (no silent degradation)
- Engines declare supported versions
- Rust validator rejects incompatible versions

## Debugging

IntentFrame can be serialized to JSON for logging:

```cpp
char* json = intent_frame_to_json(&frame);
printf("IntentFrame: %s\n", json);
free(json);
```

This allows replaying intent for debugging and testing.
