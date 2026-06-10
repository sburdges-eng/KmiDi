/**
 * intent_ir_ffi.cpp - Stateful Intent IR Manager
 *
 * Implements the stateful C API (initialize, store, get_snapshot, etc.)
 * using Rust's functional primitives (validate, clamp, builder) for
 * the heavy lifting.
 *
 * State management is C++ (thread-safe via atomics).
 * Validation and clamping delegate to Rust libintent_ir.a.
 */

#include "intent_ir_ffi.h"
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>

namespace {
// Portable 64-bit random for session ids (control plane only).
// arc4random is BSD/macOS-only; std::random_device is rand_s-backed on
// Windows and /dev/urandom-backed on Linux/macOS. Not an RT path.
uint64_t kmidi_random_session_u64() {
  std::random_device rd;
  return (static_cast<uint64_t>(rd()) << 32) | static_cast<uint64_t>(rd());
}
} // namespace

// Rust FFI imports (from libintent_ir.a)
extern "C" {
int validate_intent_frame_ffi(const CIntentFrame *frame);
void clamp_intent_frame_ffi(CIntentFrame *frame);
bool intent_frame_version_supported_ffi(uint16_t version);
}

// ============================================================================
// Internal state
// ============================================================================

namespace {

struct IntentIRState {
  std::mutex mutex;
  CIntentFrame currentFrame{};
  std::atomic<uint64_t> nextIntentId{1};
  std::atomic<uint64_t> sessionId{0};
  bool initialized{false};
  const char *lastError{nullptr};
};

// Global singleton — safe because all access is mutex-protected
static IntentIRState g_state;

// Map Rust validation error codes to our error codes
IntentIRErrorCode mapRustError(int rustCode) {
  switch (rustCode) {
  case 0:
    return INTENT_IR_SUCCESS;
  case 1:
    return INTENT_IR_ERROR_INVALID_VERSION;
  case 2: // InvalidValence
  case 3: // InvalidArousal
  case 4: // InvalidDominance
  case 5: // InvalidIntensity
  case 6: // InvalidConfidence
    return INTENT_IR_ERROR_OUT_OF_RANGE;
  case 7: // InvalidTempoBias
  case 8: // InvalidDensity
    return INTENT_IR_ERROR_OUT_OF_RANGE;
  case 9:
    return INTENT_IR_ERROR_INVALID_DISCRETE_ID;
  case 10:
    return INTENT_IR_ERROR_INVALID_TIME_SCOPE;
  case 11:
    return INTENT_IR_ERROR_INVALID_CONSTRAINTS;
  default:
    return INTENT_IR_ERROR_UNKNOWN;
  }
}

// Static error messages
const char *errorMessages[] = {
    "Success",                     // 0
    "Invalid IR version",          // -1
    "Value out of range",          // -2
    "Invalid discrete emotion ID", // -3
    "Invalid time scope",          // -4
    "Invalid constraints",         // -5
    "Null pointer",                // -6
    "Memory allocation error",     // -7
};

} // anonymous namespace

// ============================================================================
// Public API implementation
// ============================================================================

extern "C" {

IntentIRErrorCode intent_ir_initialize(uint64_t session_id) {
  std::lock_guard<std::mutex> lock(g_state.mutex);

  if (session_id == 0) {
    // Auto-generate session ID from time + random
    session_id = kmidi_random_session_u64();
  }

  g_state.sessionId.store(session_id, std::memory_order_release);
  g_state.nextIntentId.store(1, std::memory_order_release);

  // Initialize default frame
  std::memset(&g_state.currentFrame, 0, sizeof(CIntentFrame));
  g_state.currentFrame.meta.ir_version = INTENT_IR_VERSION;
  g_state.currentFrame.meta.session_id = session_id;
  g_state.currentFrame.meta.intent_id = 0;

  // Set sensible defaults
  g_state.currentFrame.emotion.arousal = 0.5f;
  g_state.currentFrame.emotion.dominance = 0.5f;
  g_state.currentFrame.emotion.discrete_id = -1;
  g_state.currentFrame.emotion.confidence = 1.0f;

  g_state.currentFrame.music.rhythmic_density = 0.5f;
  g_state.currentFrame.music.groove_strength = 0.5f;
  g_state.currentFrame.music.harmonic_tension = 0.5f;
  g_state.currentFrame.music.harmonic_motion = 0.5f;
  g_state.currentFrame.music.melodic_activity = 0.5f;
  g_state.currentFrame.music.contour_variance = 0.5f;
  g_state.currentFrame.music.dynamic_range = 0.5f;
  g_state.currentFrame.music.texture_density = 0.5f;

  g_state.currentFrame.time.start_bar = -1;
  g_state.currentFrame.time.end_bar = -1;

  g_state.currentFrame.constraints.allowed_engines_mask = 0xFFFFFFFF;
  g_state.currentFrame.constraints.max_cpu_cost = 1.0f;
  g_state.currentFrame.constraints.max_event_rate = 1000.0f;

  g_state.currentFrame.provenance.user_override_weight = 0.5f;

  g_state.initialized = true;
  g_state.lastError = nullptr;

  return INTENT_IR_SUCCESS;
}

IntentIRErrorCode intent_ir_validate_and_store(const CIntentFrame *frame) {
  if (!frame) {
    g_state.lastError = "Null frame pointer";
    return INTENT_IR_ERROR_NULL_POINTER;
  }

  // Delegate validation to Rust
  int rustResult = validate_intent_frame_ffi(frame);
  if (rustResult != 0) {
    IntentIRErrorCode code = mapRustError(rustResult);
    g_state.lastError = intent_ir_get_error_message(code);
    return code;
  }

  std::lock_guard<std::mutex> lock(g_state.mutex);
  g_state.currentFrame = *frame;
  g_state.sessionId.store(frame->meta.session_id, std::memory_order_release);
  g_state.lastError = nullptr;
  return INTENT_IR_SUCCESS;
}

void intent_ir_clamp_and_store(const CIntentFrame *frame) {
  if (!frame)
    return;

  CIntentFrame clamped = *frame;
  // Delegate clamping to Rust
  clamp_intent_frame_ffi(&clamped);

  std::lock_guard<std::mutex> lock(g_state.mutex);
  g_state.currentFrame = clamped;
  g_state.sessionId.store(clamped.meta.session_id, std::memory_order_release);
  g_state.lastError = nullptr;
}

IntentIRErrorCode intent_ir_get_snapshot(CIntentFrame *frame_out) {
  if (!frame_out) {
    g_state.lastError = "Null output pointer";
    return INTENT_IR_ERROR_NULL_POINTER;
  }

  std::lock_guard<std::mutex> lock(g_state.mutex);
  *frame_out = g_state.currentFrame;
  return INTENT_IR_SUCCESS;
}

IntentIRErrorCode intent_ir_get_c_snapshot(CIntentFrame *frame_out) {
  return intent_ir_get_snapshot(frame_out);
}

bool intent_ir_is_valid(void) {
  std::lock_guard<std::mutex> lock(g_state.mutex);
  return g_state.currentFrame.meta.intent_id != 0;
}

uint64_t intent_ir_get_current_intent_id(void) {
  std::lock_guard<std::mutex> lock(g_state.mutex);
  return g_state.currentFrame.meta.intent_id;
}

uint64_t intent_ir_get_current_session_id(void) {
  std::lock_guard<std::mutex> lock(g_state.mutex);
  return g_state.currentFrame.meta.session_id;
}

uint64_t intent_ir_new_session_id(void) {
  uint64_t id = kmidi_random_session_u64();
  std::lock_guard<std::mutex> lock(g_state.mutex);
  g_state.sessionId.store(id, std::memory_order_release);
  g_state.currentFrame.meta.session_id = id;
  g_state.lastError = nullptr;
  return id;
}

uint64_t intent_ir_new_intent_id(void) {
  return g_state.nextIntentId.fetch_add(1, std::memory_order_acq_rel);
}

IntentIRErrorCode intent_ir_update_emotion(float valence, float arousal,
                                           float dominance, int16_t discrete_id,
                                           float intensity, float confidence) {
  // Copy the frame under the lock, release, then validate outside the lock
  // to avoid holding the mutex across the Rust FFI call (prevents deadlock
  // if Rust validation ever calls back into intent_ir_* functions).
  decltype(g_state.currentFrame) frame_copy;
  {
    std::lock_guard<std::mutex> lock(g_state.mutex);

    g_state.currentFrame.emotion.valence = valence;
    g_state.currentFrame.emotion.arousal = arousal;
    g_state.currentFrame.emotion.dominance = dominance;
    g_state.currentFrame.emotion.discrete_id = discrete_id;
    g_state.currentFrame.emotion.intensity = intensity;
    g_state.currentFrame.emotion.confidence = confidence;
    g_state.currentFrame.meta.intent_id =
        g_state.nextIntentId.fetch_add(1, std::memory_order_acq_rel);

    frame_copy = g_state.currentFrame;
  } // lock released before Rust call

  // Validate via Rust (outside mutex)
  int rustResult = validate_intent_frame_ffi(&frame_copy);
  if (rustResult != 0) {
    clamp_intent_frame_ffi(&frame_copy);
    // Write clamped result back under the lock
    std::lock_guard<std::mutex> lock(g_state.mutex);
    g_state.currentFrame.emotion = frame_copy.emotion;
  }

  return INTENT_IR_SUCCESS;
}

IntentIRErrorCode
intent_ir_update_music(float tempo_bias, float rhythmic_density,
                       float groove_strength, float harmonic_tension,
                       float harmonic_motion, int8_t mode_preference,
                       float melodic_activity, float contour_variance,
                       float dynamic_range, float texture_density) {
  decltype(g_state.currentFrame) frame_copy;
  {
    std::lock_guard<std::mutex> lock(g_state.mutex);

    g_state.currentFrame.music.tempo_bias = tempo_bias;
    g_state.currentFrame.music.rhythmic_density = rhythmic_density;
    g_state.currentFrame.music.groove_strength = groove_strength;
    g_state.currentFrame.music.harmonic_tension = harmonic_tension;
    g_state.currentFrame.music.harmonic_motion = harmonic_motion;
    g_state.currentFrame.music.mode_preference = mode_preference;
    g_state.currentFrame.music.melodic_activity = melodic_activity;
    g_state.currentFrame.music.contour_variance = contour_variance;
    g_state.currentFrame.music.dynamic_range = dynamic_range;
    g_state.currentFrame.music.texture_density = texture_density;
    g_state.currentFrame.meta.intent_id =
        g_state.nextIntentId.fetch_add(1, std::memory_order_acq_rel);

    frame_copy = g_state.currentFrame;
  }

  int rustResult = validate_intent_frame_ffi(&frame_copy);
  if (rustResult != 0) {
    clamp_intent_frame_ffi(&frame_copy);
    std::lock_guard<std::mutex> lock(g_state.mutex);
    g_state.currentFrame.music = frame_copy.music;
  }

  return INTENT_IR_SUCCESS;
}

const char *intent_ir_get_error_message(IntentIRErrorCode error_code) {
  int idx = -static_cast<int>(error_code);
  if (idx >= 0 && idx < 8) {
    return errorMessages[idx];
  }
  return "Unknown error";
}

const char *intent_ir_get_last_error(void) {
  return g_state.lastError ? g_state.lastError : "No error";
}

} // extern "C"
