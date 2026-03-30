#pragma once

/**
 * IntentFrame.h - C++ Intent IR Consumer
 * 
 * This header provides C struct definitions matching the Rust Intent IR v1 spec.
 * These structs are used for FFI communication and audio-thread-safe access.
 * 
 * Rules:
 * - No allocation
 * - No interpretation
 * - Audio-thread-safe accessors only
 */

#include <cstdint>
#include <cstddef>

namespace kelly {
namespace intent_ir {

// Forward declarations
struct IntentMeta;
struct EmotionState;
struct MusicalIntent;
struct TimeScope;
struct IntentConstraints;
struct IntentProvenance;
struct IntentFrame;

/**
 * IntentMeta - Routing, compatibility, debugging
 * Must match Rust CIntentMeta layout exactly
 */
struct IntentMeta {
    uint16_t ir_version;    // IR version (e.g., 1)
    uint64_t intent_id;     // Stable hash for this intent
    uint64_t session_id;    // Session-scoped ID
};

/**
 * EmotionState - VAD coordinates with optional discrete mapping
 * Must match Rust CEmotionState layout exactly
 */
struct EmotionState {
    float valence;          // [-1.0, 1.0]
    float arousal;          // [0.0, 1.0]
    float dominance;        // [0.0, 1.0]
    int16_t discrete_id;    // -1 if unused, else EmotionThesaurus ID
    float intensity;        // [0.0, 1.0]
    float confidence;       // [0.0, 1.0]
};

/**
 * MusicalIntent - Biases and tendencies (no notes, no MIDI)
 * Must match Rust CMusicalIntent layout exactly
 */
struct MusicalIntent {
    float tempo_bias;           // [-1.0, 1.0]
    float rhythmic_density;     // [0.0, 1.0]
    float groove_strength;       // [0.0, 1.0]
    float harmonic_tension;      // [0.0, 1.0]
    float harmonic_motion;       // [0.0, 1.0]
    int8_t mode_preference;      // -1 (minor), 0 (neutral), +1 (major)
    float melodic_activity;      // [0.0, 1.0]
    float contour_variance;      // [0.0, 1.0]
    float dynamic_range;         // [0.0, 1.0]
    float texture_density;       // [0.0, 1.0]
};

/**
 * TimeScope - Intent without time is noise
 * Must match Rust CTimeScope layout exactly
 */
struct TimeScope {
    int32_t start_bar;      // Inclusive, -1 = immediate
    int32_t end_bar;        // Exclusive, -1 = open-ended
    float fade_in_beats;    // 0.0 = hard
    float fade_out_beats;   // 0.0 = hard
};

/**
 * IntentConstraints - Limit generation, not force it
 * Must match Rust CIntentConstraints layout exactly
 */
struct IntentConstraints {
    uint32_t allowed_engines_mask;    // Bitmask of allowed engines
    uint32_t forbidden_engines_mask;  // Bitmask of forbidden engines
    float max_cpu_cost;               // Hint, not guarantee
    float max_event_rate;              // Max event rate
};

/**
 * IntentSource - Provenance tracking
 * Must match Rust CIntentSource layout exactly
 */
enum class IntentSource : uint8_t {
    UiDirect = 0,
    UiEdit = 1,
    MlText = 2,
    MlAudio = 3,
    Preset = 4,
    Automation = 5,
};

/**
 * IntentProvenance - Debugging and trust
 * Must match Rust CIntentProvenance layout exactly
 */
struct IntentProvenance {
    IntentSource source;               // Source of this intent
    float user_override_weight;        // [0.0, 1.0] - 0.0 = ML dominates, 1.0 = user dominates
};

/**
 * IntentFrame - Top-level unit representing one musical intention
 * Must match Rust CIntentFrame layout exactly
 */
struct IntentFrame {
    IntentMeta meta;
    EmotionState emotion;
    MusicalIntent music;
    TimeScope time;
    IntentConstraints constraints;
    IntentProvenance provenance;
};

/**
 * IntentFrameSnapshot - Thread-safe snapshot for audio thread consumption
 * 
 * This is a copy of IntentFrame that can be safely read from the audio thread.
 * The snapshot is updated atomically by the Rust boundary layer.
 */
class IntentFrameSnapshot {
public:
    IntentFrameSnapshot() = default;
    explicit IntentFrameSnapshot(const IntentFrame& frame);
    
    // Audio-thread-safe accessors (read-only)
    const IntentMeta& getMeta() const { return frame_.meta; }
    const EmotionState& getEmotion() const { return frame_.emotion; }
    const MusicalIntent& getMusic() const { return frame_.music; }
    const TimeScope& getTime() const { return frame_.time; }
    const IntentConstraints& getConstraints() const { return frame_.constraints; }
    const IntentProvenance& getProvenance() const { return frame_.provenance; }
    
    // Get full frame (read-only)
    const IntentFrame& getFrame() const { return frame_; }
    
    // Update snapshot (called from non-audio thread only)
    void update(const IntentFrame& frame);
    
    // Check if snapshot is valid (non-zero intent_id)
    bool isValid() const { return frame_.meta.intent_id != 0; }
    
private:
    IntentFrame frame_;
};

/**
 * Helper functions for reading IntentFrame
 * These are pure functions with no side effects, safe for audio thread
 */

// Get VAD values from emotion state
inline void getVAD(const EmotionState& emotion, float& valence, float& arousal, float& dominance) {
    valence = emotion.valence;
    arousal = emotion.arousal;
    dominance = emotion.dominance;
}

// Check if discrete emotion is set
inline bool hasDiscreteEmotion(const EmotionState& emotion) {
    return emotion.discrete_id >= 0;
}

// Check if time scope is immediate
inline bool isImmediate(const TimeScope& time) {
    return time.start_bar == -1;
}

// Check if time scope is open-ended
inline bool isOpenEnded(const TimeScope& time) {
    return time.end_bar == -1;
}

// Check if engine is allowed
inline bool isEngineAllowed(const IntentConstraints& constraints, uint32_t engine_mask) {
    return (constraints.allowed_engines_mask & engine_mask) != 0 &&
           (constraints.forbidden_engines_mask & engine_mask) == 0;
}

} // namespace intent_ir
} // namespace kelly
