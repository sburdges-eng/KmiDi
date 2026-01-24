#pragma once

/**
 * EngineContract.h - Engine interpretation rules for Intent IR
 *
 * This file documents which engines consume which IR fields and how they
 * interpret them. No engine is allowed to invent meaning - all interpretation
 * must be documented here.
 */

#include "IntentFrame.h"
#include <cstdint>

namespace kelly {
namespace intent_ir {

/**
 * Engine IDs for bitmask operations
 */
enum class EngineId : uint32_t {
  None = 0,
  VADSystem = 1 << 0,
  EmotionToMusicMapper = 1 << 1,
  IntentPipeline = 1 << 2,
  MidiGenerator = 1 << 3,
  GrooveEngine = 1 << 4,
  HarmonyEngine = 1 << 5,
  All = UINT32_MAX,
};

/**
 * Engine Contract - Documents which engines use which IR fields
 */
struct EngineContract {
  // VADSystem consumes:
  // - emotion.valence, emotion.arousal, emotion.dominance
  // - emotion.discrete_id (if >= 0)
  // - emotion.confidence (for smoothing)
  // Does NOT consume: music.*, time.*, constraints.*

  // EmotionToMusicMapper consumes:
  // - emotion.valence, emotion.arousal, emotion.dominance
  // - music.* (all fields)
  // Does NOT consume: time.*, constraints.*

  // IntentPipeline consumes:
  // - emotion.* (all fields)
  // - music.* (all fields)
  // - time.* (for scheduling)
  // - constraints.* (for engine selection)
  // Does NOT invent: any new meaning

  // MidiGenerator consumes:
  // - music.* (all fields)
  // - time.* (for note timing)
  // - constraints.max_event_rate (for throttling)
  // Does NOT consume: emotion.* (emotion already mapped to music)

  // GrooveEngine consumes:
  // - music.rhythmic_density
  // - music.groove_strength
  // - music.tempo_bias
  // Does NOT consume: emotion.*, harmony fields

  // HarmonyEngine consumes:
  // - music.harmonic_tension
  // - music.harmonic_motion
  // - music.mode_preference
  // Does NOT consume: emotion.*, rhythm fields
};

/**
 * Helper to check if an engine should process a given IntentFrame
 */
inline bool shouldEngineProcess(EngineId engine, const IntentFrame &frame) {
  uint32_t engine_mask = static_cast<uint32_t>(engine);
  return isEngineAllowed(frame.constraints, engine_mask);
}

} // namespace intent_ir
} // namespace kelly
