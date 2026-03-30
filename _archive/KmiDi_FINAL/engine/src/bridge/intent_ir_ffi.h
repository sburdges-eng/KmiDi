#pragma once

/**
 * intent_ir_ffi.h - C FFI interface for Intent IR
 * 
 * This header provides C-compatible functions for interacting with the Rust
 * Intent IR Manager. All functions are thread-safe and can be called from
 * C++ code.
 */

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct CIntentFrame;
struct CIntentMeta;
struct CEmotionState;
struct CMusicalIntent;
struct CTimeScope;
struct CIntentConstraints;
struct CIntentProvenance;

/**
 * Error codes
 */
typedef enum {
    INTENT_IR_SUCCESS = 0,
    INTENT_IR_ERROR_INVALID_VERSION = -1,
    INTENT_IR_ERROR_OUT_OF_RANGE = -2,
    INTENT_IR_ERROR_INVALID_DISCRETE_ID = -3,
    INTENT_IR_ERROR_INVALID_TIME_SCOPE = -4,
    INTENT_IR_ERROR_INVALID_CONSTRAINTS = -5,
    INTENT_IR_ERROR_NULL_POINTER = -6,
    INTENT_IR_ERROR_MEMORY = -7,
    INTENT_IR_ERROR_UNKNOWN = -999,
} IntentIRErrorCode;

/**
 * Intent IR version
 */
#define INTENT_IR_VERSION 1

/**
 * C-compatible IntentFrame structure
 * Must match Rust CIntentFrame layout exactly
 */
typedef struct CIntentMeta {
    uint16_t ir_version;
    uint64_t intent_id;
    uint64_t session_id;
} CIntentMeta;

typedef struct CEmotionState {
    float valence;
    float arousal;
    float dominance;
    int16_t discrete_id;
    float intensity;
    float confidence;
} CEmotionState;

typedef struct CMusicalIntent {
    float tempo_bias;
    float rhythmic_density;
    float groove_strength;
    float harmonic_tension;
    float harmonic_motion;
    int8_t mode_preference;
    float melodic_activity;
    float contour_variance;
    float dynamic_range;
    float texture_density;
} CMusicalIntent;

typedef struct CTimeScope {
    int32_t start_bar;
    int32_t end_bar;
    float fade_in_beats;
    float fade_out_beats;
} CTimeScope;

typedef struct CIntentConstraints {
    uint32_t allowed_engines_mask;
    uint32_t forbidden_engines_mask;
    float max_cpu_cost;
    float max_event_rate;
} CIntentConstraints;

typedef enum {
    C_INTENT_SOURCE_UI_DIRECT = 0,
    C_INTENT_SOURCE_UI_EDIT = 1,
    C_INTENT_SOURCE_ML_TEXT = 2,
    C_INTENT_SOURCE_ML_AUDIO = 3,
    C_INTENT_SOURCE_PRESET = 4,
    C_INTENT_SOURCE_AUTOMATION = 5,
} CIntentSource;

typedef struct CIntentProvenance {
    CIntentSource source;
    float user_override_weight;
} CIntentProvenance;

typedef struct CIntentFrame {
    CIntentMeta meta;
    CEmotionState emotion;
    CMusicalIntent music;
    CTimeScope time;
    CIntentConstraints constraints;
    CIntentProvenance provenance;
} CIntentFrame;

/**
 * Initialize Intent IR Manager
 * 
 * @param session_id Optional session ID (0 = auto-generate)
 * @return Error code
 */
IntentIRErrorCode intent_ir_initialize(uint64_t session_id);

/**
 * Validate and store an IntentFrame
 * 
 * @param frame IntentFrame to validate and store
 * @return Error code
 */
IntentIRErrorCode intent_ir_validate_and_store(const CIntentFrame* frame);

/**
 * Clamp and store an IntentFrame (always succeeds)
 * 
 * @param frame IntentFrame to clamp and store
 */
void intent_ir_clamp_and_store(const CIntentFrame* frame);

/**
 * Get current IntentFrame snapshot
 * 
 * @param frame_out Output buffer for IntentFrame
 * @return Error code
 */
IntentIRErrorCode intent_ir_get_snapshot(CIntentFrame* frame_out);

/**
 * Get current IntentFrame snapshot (C struct version for C++)
 * 
 * @param frame_out Output buffer for IntentFrame
 * @return Error code
 */
IntentIRErrorCode intent_ir_get_c_snapshot(CIntentFrame* frame_out);

/**
 * Check if current frame is valid
 * 
 * @return true if valid (non-zero intent_id)
 */
bool intent_ir_is_valid(void);

/**
 * Get current intent ID
 * 
 * @return Current intent ID
 */
uint64_t intent_ir_get_current_intent_id(void);

/**
 * Get current session ID
 * 
 * @return Current session ID
 */
uint64_t intent_ir_get_current_session_id(void);

/**
 * Generate new session ID
 * 
 * @return New session ID
 */
uint64_t intent_ir_new_session_id(void);

/**
 * Generate new intent ID
 * 
 * @return New intent ID
 */
uint64_t intent_ir_new_intent_id(void);

/**
 * Update emotion state
 * 
 * @param valence Valence value [-1.0, 1.0]
 * @param arousal Arousal value [0.0, 1.0]
 * @param dominance Dominance value [0.0, 1.0]
 * @param discrete_id Discrete emotion ID (-1 if unused)
 * @param intensity Intensity [0.0, 1.0]
 * @param confidence Confidence [0.0, 1.0]
 * @return Error code
 */
IntentIRErrorCode intent_ir_update_emotion(
    float valence,
    float arousal,
    float dominance,
    int16_t discrete_id,
    float intensity,
    float confidence
);

/**
 * Update musical intent
 * 
 * @param tempo_bias Tempo bias [-1.0, 1.0]
 * @param rhythmic_density Rhythmic density [0.0, 1.0]
 * @param groove_strength Groove strength [0.0, 1.0]
 * @param harmonic_tension Harmonic tension [0.0, 1.0]
 * @param harmonic_motion Harmonic motion [0.0, 1.0]
 * @param mode_preference Mode preference [-1, 1]
 * @param melodic_activity Melodic activity [0.0, 1.0]
 * @param contour_variance Contour variance [0.0, 1.0]
 * @param dynamic_range Dynamic range [0.0, 1.0]
 * @param texture_density Texture density [0.0, 1.0]
 * @return Error code
 */
IntentIRErrorCode intent_ir_update_music(
    float tempo_bias,
    float rhythmic_density,
    float groove_strength,
    float harmonic_tension,
    float harmonic_motion,
    int8_t mode_preference,
    float melodic_activity,
    float contour_variance,
    float dynamic_range,
    float texture_density
);

/**
 * Get error message for error code
 * 
 * @param error_code Error code
 * @return Error message (static string, do not free)
 */
const char* intent_ir_get_error_message(IntentIRErrorCode error_code);

/**
 * Get last error message
 * 
 * @return Last error message (static string, do not free)
 */
const char* intent_ir_get_last_error(void);

#ifdef __cplusplus
}
#endif
