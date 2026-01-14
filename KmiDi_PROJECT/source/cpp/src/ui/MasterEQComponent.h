#pragma once

#include "ui/EQCurveView.h"
#include "ui/EQBandControls.h"
#include "ui/AIEQSuggestionEngine.h"
#include "plugin/MasterEQProcessor.h"
#include "KellyML/EmotionState.h"
#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include <array>
#include <memory>
#include <vector>

namespace kelly {

/**
 * MasterEQComponent - Main EQ component integrating curve view, band controls, and AI assist
 *
 * This component:
 * - Displays EQ curve visualization
 * - Provides band controls (knobs/sliders)
 * - Handles AI EQ suggestions
 * - Connects to APVTS for parameter automation
 */
class MasterEQComponent : public juce::Component,
                          public juce::AudioProcessorValueTreeState::Listener,
                          public juce::Timer {
public:
  MasterEQComponent(juce::AudioProcessorValueTreeState &apvts,
                    MasterEQProcessor &eqProcessor);
  ~MasterEQComponent() override;

  void paint(juce::Graphics &g) override;
  void resized() override;

  // AudioProcessorValueTreeState::Listener
  void parameterChanged(const juce::String &parameterID, float newValue) override;

  // Timer (for curve updates)
  void timerCallback() override;

  /**
   * Update emotion state for AI suggestions
   * Called from UI when emotion state changes
   */
  void updateEmotionState(const kelly::ml::EmotionState &emotionState);

  /**
   * Apply AI suggested curve to user parameters
   * Called when user clicks "Apply Suggested Curve" button
   */
  void applySuggestedCurve();

private:
  juce::AudioProcessorValueTreeState &apvts_;
  MasterEQProcessor &eqProcessor_;
  AIEQSuggestionEngine aiEngine_;

  // UI Components
  std::unique_ptr<EQCurveView> curveView_;
  std::vector<std::unique_ptr<EQBandControls>> bandControls_;

  // AI Controls
  std::unique_ptr<juce::ToggleButton> aiEnabledButton_;
  std::unique_ptr<juce::Slider> aiIntensitySlider_;
  std::unique_ptr<juce::ToggleButton> lockUserBandsButton_;
  std::unique_ptr<juce::TextButton> applySuggestedButton_;

  // Parameter attachments
  std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> aiEnabledAttachment_;
  std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> aiIntensityAttachment_;
  std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> lockUserBandsAttachment_;

  // Labels
  juce::Label aiControlsLabel_;
  juce::Label aiIntensityLabel_;

  // Current state
  kelly::ml::EmotionState currentEmotionState_{};
  bool aiEnabled_ = false;
  float aiIntensity_ = 0.5f;
  bool lockUserBands_ = false;

  // Cached curves (computed from parameters)
  std::array<float, AIEQSuggestionEngine::CURVE_POINTS> userCurve_{};
  std::array<float, AIEQSuggestionEngine::CURVE_POINTS> aiSuggestedCurve_{};
  bool userCurveValid_ = false;
  bool aiCurveValid_ = false;

  /**
   * Recompute user curve from current EQ parameters
   */
  void updateUserCurve();

  /**
   * Recompute AI suggested curve
   */
  void updateAICurve();

  /**
   * Compute EQ response at a frequency from band parameters
   */
  float computeEQResponse(float freq, const MasterEQProcessor::EQState &state) const;
};

} // namespace kelly
