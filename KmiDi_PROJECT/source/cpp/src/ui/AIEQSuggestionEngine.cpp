#include "ui/AIEQSuggestionEngine.h"
#include <juce_core/juce_core.h>
#include <algorithm>
#include <cmath>

namespace kelly {

// Log-spaced frequency grid (20Hz to 20000Hz)
static const std::array<float, AIEQSuggestionEngine::CURVE_POINTS> createFrequencyGrid() {
  std::array<float, AIEQSuggestionEngine::CURVE_POINTS> grid;
  constexpr float fMin = 20.0f;
  constexpr float fMax = 20000.0f;
  constexpr int N = AIEQSuggestionEngine::CURVE_POINTS;

  for (int i = 0; i < N; ++i) {
    // Log spacing: f[i] = fMin * (fMax / fMin)^(i / (N-1))
    float ratio = static_cast<float>(i) / static_cast<float>(N - 1);
    grid[i] = fMin * std::pow(fMax / fMin, ratio);
  }

  return grid;
}

const std::array<float, AIEQSuggestionEngine::CURVE_POINTS> &
AIEQSuggestionEngine::getFrequencyGrid() {
  static const auto grid = createFrequencyGrid();
  return grid;
}

AIEQSuggestionEngine::AIEQSuggestionEngine() = default;

std::array<float, AIEQSuggestionEngine::CURVE_POINTS> AIEQSuggestionEngine::generateSuggestion(
    const kelly::ml::EmotionState &emotionState,
    const std::array<float, CURVE_POINTS> &userCurve,
    const MasterEQProcessor::EQState &userBands, float intensity, bool lockUserBands) {

  // Step 1: Generate base AI curve from emotion
  auto aiCurve = generateBaseAICurve(emotionState);

  // Step 2: Calculate deltas (AI - User)
  std::array<float, CURVE_POINTS> deltas;
  for (int i = 0; i < CURVE_POINTS; ++i) {
    deltas[i] = aiCurve[i] - userCurve[i];
  }

  // Step 3: Clamp deltas to safe mastering range
  for (auto &delta : deltas) {
    delta = clampDelta(delta, DEFAULT_MAX_DELTA_DB);
  }

  // Step 4: Apply user mask if enabled
  if (lockUserBands) {
    const auto &freqGrid = getFrequencyGrid();
    for (int i = 0; i < CURVE_POINTS; ++i) {
      float mask = calculateUserMask(freqGrid[i], userBands);
      deltas[i] *= (1.0f - mask); // Reduce AI influence where user has bands
    }
  }

  // Step 5: Blend: Final = User + intensity * clamped_delta
  std::array<float, CURVE_POINTS> result;
  for (int i = 0; i < CURVE_POINTS; ++i) {
    result[i] = userCurve[i] + intensity * deltas[i];
  }

  return result;
}

std::array<float, AIEQSuggestionEngine::CURVE_POINTS>
AIEQSuggestionEngine::generateBaseAICurve(const kelly::ml::EmotionState &emotionState) {
  std::array<float, CURVE_POINTS> curve{};
  const auto &freqGrid = getFrequencyGrid();

  for (int i = 0; i < CURVE_POINTS; ++i) {
    curve[i] = emotionToEQDelta(freqGrid[i], emotionState);
  }

  return curve;
}

float AIEQSuggestionEngine::calculateUserMask(float freq,
                                               const MasterEQProcessor::EQState &userBands) const {
  float totalMask = 0.0f;

  // For each enabled user band, calculate distance-based mask
  for (int bandIdx = 0; bandIdx < 6; ++bandIdx) {
    if (!userBands.bands[bandIdx].enabled) {
      continue;
    }

    float bandFreq = userBands.bands[bandIdx].freq;
    float bandQ = userBands.bands[bandIdx].q;

    // Calculate bandwidth (approximate)
    float bandwidthHz = bandFreq / bandQ;

    // Distance from frequency to band center
    float distance = std::abs(freq - bandFreq);

    // Gaussian mask: exp(-(distance / bandwidth)^2)
    // Closer to band = more masking (less AI influence)
    float normalizedDistance = distance / (bandwidthHz * 2.0f); // 2x bandwidth for wider protection
    float mask = std::exp(-normalizedDistance * normalizedDistance);

    // Weight by band gain (more gain = more user intent = more masking)
    float gainWeight = std::abs(userBands.bands[bandIdx].gain) / 24.0f; // Normalize to 0-1
    mask *= (0.5f + gainWeight * 0.5f); // Scale 0.5-1.0 based on gain

    totalMask = std::max(totalMask, mask);
  }

  return totalMask;
}

float AIEQSuggestionEngine::emotionToEQDelta(float freq,
                                              const kelly::ml::EmotionState &emotionState) const {
  float delta = 0.0f;

  // Emotion → EQ Mapping (advisory, not absolute)
  // These are gentle suggestions that bias the curve

  // Negative valence → soften high mids (2-4kHz), gentle low-mid boost
  if (emotionState.valence < 0.0f) {
    // High mids reduction
    if (freq >= 2000.0f && freq <= 4000.0f) {
      float factor = std::sin((freq - 2000.0f) / 2000.0f * juce::MathConstants<float>::pi);
      delta -= factor * std::abs(emotionState.valence) * 1.0f; // Max -1dB
    }

    // Low-mid gentle boost (200-800Hz)
    if (freq >= 200.0f && freq <= 800.0f) {
      float factor = std::sin((freq - 200.0f) / 600.0f * juce::MathConstants<float>::pi);
      delta += factor * std::abs(emotionState.valence) * 0.5f; // Max +0.5dB
    }
  }

  // High arousal → tighten low end (80-120Hz), reduce mud (200-400Hz)
  if (emotionState.arousal > 0.5f) {
    // Low end tightening (gentle boost at 80-120Hz)
    if (freq >= 80.0f && freq <= 120.0f) {
      float factor = std::sin((freq - 80.0f) / 40.0f * juce::MathConstants<float>::pi);
      delta += factor * (emotionState.arousal - 0.5f) * 2.0f * 0.8f; // Max +0.8dB
    }

    // Reduce mud (200-400Hz gentle cut)
    if (freq >= 200.0f && freq <= 400.0f) {
      float factor = std::sin((freq - 200.0f) / 200.0f * juce::MathConstants<float>::pi);
      delta -= factor * (emotionState.arousal - 0.5f) * 2.0f * 0.6f; // Max -0.6dB
    }
  }

  // High complexity → reduce narrow resonances, gentle broadband shaping
  if (emotionState.complexity > 0.6f) {
    // Subtle broadband reduction (less aggressive EQ)
    // This manifests as slight reduction across spectrum
    float complexityFactor = (emotionState.complexity - 0.6f) / 0.4f; // 0.0-1.0
    delta -= complexityFactor * 0.3f; // Slight broadband reduction
  }

  // Low dominance → subtle high-frequency roll-off
  if (emotionState.dominance < 0.4f) {
    if (freq >= 8000.0f) {
      float factor = (freq - 8000.0f) / 12000.0f; // 8kHz-20kHz
      factor = std::min(factor, 1.0f);
      delta -= factor * (0.4f - emotionState.dominance) / 0.4f * 1.0f; // Max -1dB at 20kHz
    }
  }

  // Clamp to absolute maximum before returning
  return clampDelta(delta, ABSOLUTE_MAX_DELTA_DB);
}

float AIEQSuggestionEngine::clampDelta(float delta, float maxDelta) {
  return juce::jlimit(-maxDelta, maxDelta, delta);
}

} // namespace kelly
