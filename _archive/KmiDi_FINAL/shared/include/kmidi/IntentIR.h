#pragma once

/**
 * Intent IR v1 - Canonical Musical Intent Representation
 *
 * A single, canonical representation of what the user wants musically,
 * independent of UI, ML models, DSP implementation, and language runtime.
 *
 * Design Rules (non-negotiable):
 * - Serializable (C struct first, JSON second)
 * - Versioned
 * - Immutable once handed to audio
 * - No pointers to foreign memory
 * - No dynamic allocation required to consume
 * - Meaningful without ML context
 */

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define INTENT_IR_VERSION 1

// =============================================================================
// INTENT SOURCE ENUM
// =============================================================================

typedef enum {
    SOURCE_UI_DIRECT = 0,
    SOURCE_UI_EDIT = 1,
    SOURCE_ML_TEXT = 2,
    SOURCE_ML_AUDIO = 3,
    SOURCE_PRESET = 4,
    SOURCE_AUTOMATION = 5
} IntentSource;

// =============================================================================
// INTENT META
// =============================================================================

/**
 * Metadata for routing, compatibility, and debugging.
 * Ignored by engines unless explicitly needed.
 */
typedef struct {
    uint16_t ir_version;        // e.g. 1
    uint64_t intent_id;         // stable hash
    uint64_t session_id;        // session-scoped
} IntentMeta;

// =============================================================================
// EMOTION STATE
// =============================================================================

/**
 * Emotion is input, not truth.
 * Engines interpret it, they don't argue with it.
 *
 * Rules:
 * - Engines must work with only VAD
 * - Discrete emotion is a hint, never required
 * - Confidence < 0.3 should soften output, not block it
 */
typedef struct {
    // Continuous (preferred)
    float valence;    // [-1.0, 1.0]
    float arousal;   // [ 0.0, 1.0]
    float dominance;  // [ 0.0, 1.0]

    // Optional discrete mapping
    int16_t discrete_id;   // -1 if unused (EmotionThesaurus ID)
    float   intensity;     // [0.0, 1.0]

    // Confidence in this emotional reading
    float confidence;      // [0.0, 1.0]
} EmotionState;

// =============================================================================
// MUSICAL INTENT
// =============================================================================

/**
 * This is where ambiguity dies.
 * No notes. No MIDI.
 * Only biases and tendencies.
 *
 * Rules:
 * - All values are normalized
 * - Engines map these to domain-specific parameters
 * - No engine gets to invent meaning
 */
typedef struct {
    // Tempo & energy
    float tempo_bias;          // -1.0 (slower) → +1.0 (faster)
    float rhythmic_density;    // 0.0 sparse → 1.0 dense
    float groove_strength;     // 0.0 loose → 1.0 tight

    // Harmony
    float harmonic_tension;    // 0.0 consonant → 1.0 tense
    float harmonic_motion;     // 0.0 static → 1.0 active
    int8_t mode_preference;    // -1 minor, 0 neutral, +1 major

    // Melody
    float melodic_activity;    // 0.0 minimal → 1.0 active
    float contour_variance;    // 0.0 flat → 1.0 wide

    // Dynamics & texture
    float dynamic_range;       // 0.0 flat → 1.0 wide
    float texture_density;     // 0.0 thin → 1.0 thick
} MusicalIntent;

// =============================================================================
// TIME SCOPE
// =============================================================================

/**
 * Intent without time is noise.
 *
 * Rules:
 * - Audio thread only sees resolved scopes
 * - Scheduling engines own translation to samples
 */
typedef struct {
    int32_t start_bar;     // inclusive, -1 = immediate
    int32_t end_bar;       // exclusive, -1 = open-ended

    float   fade_in_beats; // 0.0 = hard
    float   fade_out_beats;
} TimeScope;

// =============================================================================
// INTENT CONSTRAINTS
// =============================================================================

/**
 * Used to limit generation, not force it.
 * This is how therapy mode and pro mode coexist without forks.
 */
typedef struct {
    uint32_t allowed_engines_mask;   // bitmask
    uint32_t forbidden_engines_mask;

    float max_cpu_cost;  // hint, not guarantee
    float max_event_rate;
} IntentConstraints;

// =============================================================================
// INTENT PROVENANCE
// =============================================================================

/**
 * This is how you debug and trust the system.
 *
 * Rules:
 * - Engines may respond differently based on source
 * - UI never lies about this
 */
typedef struct {
    IntentSource source;
    float user_override_weight; // 0.0 → ML dominates, 1.0 → user dominates
} IntentProvenance;

// =============================================================================
// INTENT FRAME (Top-Level)
// =============================================================================

/**
 * One IntentFrame represents one musical intention over a bounded scope.
 *
 * This is the only thing allowed to:
 * - cross FFI
 * - be logged
 * - be replayed
 * - be tested deterministically
 */
#pragma pack(push, 1)
typedef struct {
    IntentMeta        meta;
    EmotionState      emotion;
    MusicalIntent     music;
    TimeScope         time;
    IntentConstraints constraints;
    IntentProvenance  provenance;
} IntentFrame;
#pragma pack(pop)

// =============================================================================
// VALIDATION HELPERS
// =============================================================================

/**
 * Clamp all float values in an IntentFrame to valid ranges.
 * Called before handing to audio thread.
 * Implemented in Rust validator, called via FFI.
 */
void intent_frame_clamp(IntentFrame* frame);

/**
 * Validate an IntentFrame.
 * Returns true if valid, false otherwise.
 * Implemented in Rust validator, called via FFI.
 */
bool intent_frame_validate(const IntentFrame* frame);

/**
 * Check if version is supported.
 * Implemented in Rust validator, called via FFI.
 */
bool intent_frame_version_supported(uint16_t version);

#ifdef __cplusplus
}
#endif
