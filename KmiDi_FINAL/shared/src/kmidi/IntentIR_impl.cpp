#include "kmidi/IntentIR.h"

// Implementation stubs that call Rust validator via FFI
// These will be linked against the Rust static library

extern "C" {
    // FFI functions from Rust crate (matching ffi.rs exports)
    extern void clamp_intent_frame_ffi(IntentFrame* frame);
    extern int validate_intent_frame_ffi(const IntentFrame* frame);
    extern bool intent_frame_version_supported_ffi(uint16_t version);
}

void intent_frame_clamp(IntentFrame* frame) {
    if (frame) {
        clamp_intent_frame_ffi(frame);
    }
}

bool intent_frame_validate(const IntentFrame* frame) {
    if (!frame) {
        return false;
    }
    // Returns 0 on success, error code on failure
    return validate_intent_frame_ffi(frame) == 0;
}

bool intent_frame_version_supported(uint16_t version) {
    return intent_frame_version_supported_ffi(version);
}
