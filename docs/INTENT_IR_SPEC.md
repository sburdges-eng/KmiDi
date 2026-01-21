# Intent IR v1 Specification

## Purpose

Intent IR (Intent Intermediate Representation) is the canonical representation of musical intent, independent of UI, ML models, or DSP implementation. Everything upstream produces this. Everything downstream consumes this. Nothing bypasses it.

## Core Design Rules (Non-Negotiable)

1. **Serializable** - C struct first, JSON second
2. **Versioned** - Explicit version field, negotiation logic
3. **Immutable once handed to audio** - Audio thread only sees snapshots
4. **No pointers to foreign memory** - Self-contained
5. **No dynamic allocation required to consume** - Audio-thread-safe
6. **Meaningful without ML context** - Engines can work with VAD alone

## IntentFrame Structure

```rust
struct IntentFrame {
    meta: IntentMeta,
    emotion: EmotionState,
    music: MusicalIntent,
    time: TimeScope,
    constraints: IntentConstraints,
    provenance: IntentProvenance,
}
```

### IntentMeta

Routing, compatibility, debugging.

```rust
struct IntentMeta {
    ir_version: u16,      // IR version (e.g., 1)
    intent_id: u64,        // Stable hash for this intent
    session_id: u64,       // Session-scoped ID
}
```

### EmotionState

VAD coordinates with optional discrete mapping.

```rust
struct EmotionState {
    valence: f32,          // [-1.0, 1.0]
    arousal: f32,           // [0.0, 1.0]
    dominance: f32,         // [0.0, 1.0]
    discrete_id: i16,       // -1 if unused, else EmotionThesaurus ID
    intensity: f32,         // [0.0, 1.0]
    confidence: f32,        // [0.0, 1.0]
}
```

**Rules:**
- Engines must work with only VAD
- Discrete emotion is a hint, never required
- Confidence < 0.3 should soften output, not block it

### MusicalIntent

Biases and tendencies (no notes, no MIDI).

```rust
struct MusicalIntent {
    tempo_bias: f32,           // [-1.0, 1.0]
    rhythmic_density: f32,     // [0.0, 1.0]
    groove_strength: f32,       // [0.0, 1.0]
    harmonic_tension: f32,      // [0.0, 1.0]
    harmonic_motion: f32,       // [0.0, 1.0]
    mode_preference: i8,        // -1 (minor), 0 (neutral), +1 (major)
    melodic_activity: f32,      // [0.0, 1.0]
    contour_variance: f32,       // [0.0, 1.0]
    dynamic_range: f32,         // [0.0, 1.0]
    texture_density: f32,        // [0.0, 1.0]
}
```

**Rules:**
- All values are normalized
- Engines map these to domain-specific parameters
- No engine gets to invent meaning

### TimeScope

Intent without time is noise.

```rust
struct TimeScope {
    start_bar: i32,         // Inclusive, -1 = immediate
    end_bar: i32,           // Exclusive, -1 = open-ended
    fade_in_beats: f32,     // 0.0 = hard
    fade_out_beats: f32,    // 0.0 = hard
}
```

### IntentConstraints

Limit generation, not force it.

```rust
struct IntentConstraints {
    allowed_engines_mask: u32,      // Bitmask of allowed engines
    forbidden_engines_mask: u32,     // Bitmask of forbidden engines
    max_cpu_cost: f32,               // Hint, not guarantee
    max_event_rate: f32,             // Max event rate
}
```

### IntentProvenance

Debugging and trust.

```rust
enum IntentSource {
    UiDirect = 0,
    UiEdit = 1,
    MlText = 2,
    MlAudio = 3,
    Preset = 4,
    Automation = 5,
}

struct IntentProvenance {
    source: IntentSource,
    user_override_weight: f32,  // [0.0, 1.0] - 0.0 = ML dominates, 1.0 = user dominates
}
```

## Versioning Strategy

- `ir_version` increments on breaking change
- Engines declare supported versions
- Rust rejects mismatches loudly
- No silent degradation. Ever.

## Engine Interpretation Rules

See `src/core/intent_ir/EngineContract.h` for which engines consume which IR fields.

## Migration Guide

See `docs/DEBUGGING_GUIDE.md` for migration from old system.
