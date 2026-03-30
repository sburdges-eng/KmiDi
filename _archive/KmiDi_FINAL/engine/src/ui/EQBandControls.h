#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include <array>
#include <memory>

namespace kelly {

/**
 * EQBandControls - Control panel for EQ band parameters
 *
 * Provides knobs/sliders for:
 * - Frequency (log scale)
 * - Gain (dB scale)
 * - Q (linear scale, 0.1-10.0)
 * - Enable/disable toggle
 *
 * All controls are attached to APVTS parameters for host automation.
 */
class EQBandControls : public juce::Component {
public:
  /**
   * Band type determines which controls are shown
   */
  enum class BandType {
    LowCut,    // Frequency only, Q fixed
    LowShelf,  // Freq, Gain, Q
    Parametric,// Freq, Gain, Q
    HighShelf  // Freq, Gain, Q
  };

  EQBandControls(int bandIndex, BandType bandType,
                 juce::AudioProcessorValueTreeState &apvts);
  ~EQBandControls() override = default;

  void paint(juce::Graphics &g) override;
  void resized() override;

  /**
   * Get band type
   */
  BandType getBandType() const { return bandType_; }

  /**
   * Get band index (0-5)
   */
  int getBandIndex() const { return bandIndex_; }

private:
  int bandIndex_;
  BandType bandType_;

  // Sliders
  std::unique_ptr<juce::Slider> freqSlider_;
  std::unique_ptr<juce::Slider> gainSlider_;
  std::unique_ptr<juce::Slider> qSlider_;
  std::unique_ptr<juce::ToggleButton> enableButton_;

  // Parameter attachments
  std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> freqAttachment_;
  std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> gainAttachment_;
  std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> qAttachment_;
  std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> enableAttachment_;

  // Labels
  juce::Label freqLabel_;
  juce::Label gainLabel_;
  juce::Label qLabel_;
  juce::Label enableLabel_;
  juce::Label bandNameLabel_;

  /**
   * Get parameter ID for frequency
   */
  const char *getFreqParamID() const;

  /**
   * Get parameter ID for gain
   */
  const char *getGainParamID() const;

  /**
   * Get parameter ID for Q
   */
  const char *getQParamID() const;

  /**
   * Get parameter ID for enable
   */
  const char *getEnableParamID() const;

  /**
   * Get band display name
   */
  juce::String getBandName() const;
};

} // namespace kelly
