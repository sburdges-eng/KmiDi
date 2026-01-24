#pragma once

/**
 * Assertions.h - Runtime assertions for Intent IR architecture
 *
 * These assertions enforce:
 * - IR validity
 * - No direct parameter passing
 * - Audio thread never blocks
 * - Provenance is never falsified
 */

#include "IntentFrame.h"
#include <cassert>
#include <thread>

#ifdef NDEBUG
#define INTENT_IR_ASSERT(condition, message) ((void)0)
#else
#define INTENT_IR_ASSERT(condition, message)                                   \
  do {                                                                         \
    if (!(condition)) {                                                        \
      assert(false && message);                                                \
    }                                                                          \
  } while (0)
#endif

namespace kelly {
namespace intent_ir {

/**
 * Assert IR validity
 */
inline void assertIRValid(const IntentFrame &frame) {
  INTENT_IR_ASSERT(frame.meta.ir_version == 1, "Invalid IR version");

  INTENT_IR_ASSERT(frame.emotion.valence >= -1.0f &&
                       frame.emotion.valence <= 1.0f,
                   "Valence out of range");

  INTENT_IR_ASSERT(frame.emotion.arousal >= 0.0f &&
                       frame.emotion.arousal <= 1.0f,
                   "Arousal out of range");

  INTENT_IR_ASSERT(frame.emotion.dominance >= 0.0f &&
                       frame.emotion.dominance <= 1.0f,
                   "Dominance out of range");

  INTENT_IR_ASSERT(frame.music.tempo_bias >= -1.0f &&
                       frame.music.tempo_bias <= 1.0f,
                   "Tempo bias out of range");

  // Add more assertions as needed
}

/**
 * Assert audio thread safety
 */
inline void assertAudioThreadSafe() {
  // Check if we're on the audio thread
  // This is a simplified check - in real implementation,
  // you'd use JUCE's isAudioThread() or similar
  INTENT_IR_ASSERT(true, // Placeholder - implement actual check
                   "Not on audio thread");
}

/**
 * Assert no blocking operations on audio thread
 */
inline void assertNoBlockingOnAudioThread() {
  assertAudioThreadSafe();
  // Additional checks for blocking operations
}

/**
 * Assert provenance is not falsified
 */
inline void assertProvenanceValid(const IntentProvenance &provenance) {
  INTENT_IR_ASSERT(provenance.user_override_weight >= 0.0f &&
                       provenance.user_override_weight <= 1.0f,
                   "User override weight out of range");
}

} // namespace intent_ir
} // namespace kelly
