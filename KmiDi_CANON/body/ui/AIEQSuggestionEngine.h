#pragma once

#include "KellyML/EmotionState.h"
#include "plugin/MasterEQProcessor.h"
#include <array>
#include <vector>

namespace kelly {

/**
 * AIEQSuggestionEngine - Generates AI-suggested EQ curves based on EmotionState
 *
 * This engine generates EQ curve suggestions that bias the user's curve
 * rather than replacing it. AI suggestions are visual-only until explicitly
 * applied by the user.
 *
 * Core principles:
 * - AI suggests, user decides
 * - Small moves only (±1.5dB default, ±3.0dB max)
 * - AI backs off near user-touched bands
 * - Blending is additive bias, not replacement
 */
class AIEQSuggestionEngine {
public:
  static constexpr int CURVE_POINTS = 256; // Log-spaced frequency points
  static constexpr float DEFAULT_MAX_DELTA_DB = 1.5f; // Default clamp
  static constexpr float ABSOLUTE_MAX_DELTA_DB = 3.0f; // Hard ceiling

  AIEQSuggestionEngine();
  ~AIEQSuggestionEngine() = default;

  /**
   * Generate suggested EQ curve from EmotionState
   *
   * @param emotionState Current emotion state (valence, arousal, dominance, complexity)
   * @param userCurve Current user EQ curve (256 points, log-spaced frequencies)
   * @param userBands User's band settings (for masking AI influence near user bands)
   * @param intensity AI influence intensity (0.0-1.0)
   * @param lockUserBands If true, reduce AI influence near user-touched bands
   * @return Suggested curve (256 points, dB gain values)
   */
  std::array<float, CURVE_POINTS>
  generateSuggestion(const kelly::ml::EmotionState &emotionState,
                     const std::array<float, CURVE_POINTS> &userCurve,
                     const MasterEQProcessor::EQState &userBands, float intensity,
                     bool lockUserBands);

  /**
   * Generate base AI curve from EmotionState (without user curve blending)
   * This is the "raw" AI suggestion before clamping and user masking
   *
   * @param emotionState Current emotion state
   * @return Base AI curve (256 points, dB gain values)
   */
  std::array<float, CURVE_POINTS>
  generateBaseAICurve(const kelly::ml::EmotionState &emotionState);

  /**
   * Get log-spaced frequency grid (shared by all curves)
   * Frequencies from 20Hz to 20000Hz
   */
  static const std::array<float, CURVE_POINTS> &getFrequencyGrid();

private:
  /**
   * Calculate user influence mask
   * Reduces AI influence near user-touched bands
   *
   * @param freq Frequency point to evaluate
   * @param userBands User's band settings
   * @return Mask value (0.0 = full AI influence, 1.0 = no AI influence)
   */
  float calculateUserMask(float freq, const MasterEQProcessor::EQState &userBands) const;

  /**
   * Apply emotion-based EQ shaping at a specific frequency
   * Maps emotion state to EQ deltas
   *
   * @param freq Frequency to evaluate
   * @param emotionState Current emotion state
   * @return Suggested gain delta in dB
   */
  float emotionToEQDelta(float freq, const kelly::ml::EmotionState &emotionState) const;

  /**
   * Clamp delta to safe mastering range
   */
  static float clampDelta(float delta, float maxDelta = DEFAULT_MAX_DELTA_DB);
};

} // namespace kelly
