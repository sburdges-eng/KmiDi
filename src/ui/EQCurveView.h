#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <array>

namespace kelly {

/**
 * EQCurveView - Visualizes EQ curves (user and AI suggestions)
 *
 * Displays:
 * - User EQ curve (solid line, neutral light gray)
 * - AI suggested curve (dashed line, desaturated accent color)
 * - Frequency axis (20Hz-20kHz, log scale)
 * - Gain axis (-24dB to +24dB, linear)
 * - Grid lines for reference
 *
 * Rules:
 * - No allocations in paint()
 * - Pre-compute all curve points
 * - Redraw only on parameter change
 */
class EQCurveView : public juce::Component {
public:
  static constexpr int CURVE_POINTS = 256;

  EQCurveView();
  ~EQCurveView() override = default;

  void paint(juce::Graphics &g) override;
  void resized() override;

  /**
   * Update user curve (called when EQ parameters change)
   * @param curve 256 log-spaced frequency points with gain in dB
   */
  void updateUserCurve(const std::array<float, CURVE_POINTS> &curve);

  /**
   * Update AI suggested curve (called when AI suggestions change)
   * @param curve 256 log-spaced frequency points with gain in dB
   */
  void updateAICurve(const std::array<float, CURVE_POINTS> &curve);

  /**
   * Show/hide AI curve
   */
  void setAICurveVisible(bool visible);

  /**
   * Set AI curve intensity (for fading when disabled)
   */
  void setAICurveIntensity(float intensity); // 0.0-1.0

private:
  std::array<float, CURVE_POINTS> userCurve_{};
  std::array<float, CURVE_POINTS> aiCurve_{};
  bool aiCurveVisible_ = false;
  float aiCurveIntensity_ = 1.0f;

  // Pre-computed paths (no allocations in paint)
  juce::Path userCurvePath_;
  juce::Path aiCurvePath_;
  bool userCurvePathValid_ = false;
  bool aiCurvePathValid_ = false;

  // Grid paths (cached)
  juce::Path frequencyGridPath_;
  juce::Path gainGridPath_;
  bool gridPathValid_ = false;

  /**
   * Update curve path from data points
   */
  void updateUserCurvePath();
  void updateAICurvePath();
  void updateGridPaths();

  /**
   * Convert frequency to X coordinate (log scale)
   */
  float freqToX(float freq, float width) const;

  /**
   * Convert gain (dB) to Y coordinate (linear scale)
   */
  float gainToY(float gainDb, float height) const;

  // Drawing constants
  static constexpr float FREQ_MIN = 20.0f;
  static constexpr float FREQ_MAX = 20000.0f;
  static constexpr float GAIN_MIN = -24.0f;
  static constexpr float GAIN_MAX = 24.0f;
  static constexpr float GRID_MARGIN = 40.0f;
};

} // namespace kelly
