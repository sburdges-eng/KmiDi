#include "intent_ir_ffi.h"

// This file provides C++ wrapper functions that call the C FFI functions.
// The actual implementation is in Rust (intent_ir_ffi_exports.rs).

// For now, these are stubs that will be linked to the Rust implementation
// when the Rust library is built and linked.

extern "C" {
    // These functions are exported from the Rust library
    IntentIRErrorCode intent_ir_initialize(uint64_t session_id);
    IntentIRErrorCode intent_ir_validate_and_store(const CIntentFrame* frame);
    void intent_ir_clamp_and_store(const CIntentFrame* frame);
    IntentIRErrorCode intent_ir_get_snapshot(CIntentFrame* frame_out);
    IntentIRErrorCode intent_ir_get_c_snapshot(CIntentFrame* frame_out);
    bool intent_ir_is_valid(void);
    uint64_t intent_ir_get_current_intent_id(void);
    uint64_t intent_ir_get_current_session_id(void);
    uint64_t intent_ir_new_session_id(void);
    uint64_t intent_ir_new_intent_id(void);
    IntentIRErrorCode intent_ir_update_emotion(
        float valence,
        float arousal,
        float dominance,
        int16_t discrete_id,
        float intensity,
        float confidence
    );
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
    const char* intent_ir_get_error_message(IntentIRErrorCode error_code);
    const char* intent_ir_get_last_error(void);
}

// C++ convenience wrappers can be added here if needed
