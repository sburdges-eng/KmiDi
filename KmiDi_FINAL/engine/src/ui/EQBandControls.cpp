#include "ui/EQBandControls.h"
#include "plugin/PluginProcessor.h"

namespace kelly {

EQBandControls::EQBandControls(int bandIndex, BandType bandType,
                               juce::AudioProcessorValueTreeState &apvts)
    : bandIndex_(bandIndex), bandType_(bandType) {

  // Band name label
  bandNameLabel_.setText(getBandName(), juce::dontSendNotification);
  bandNameLabel_.setJustificationType(juce::Justification::centred);
  bandNameLabel_.setColour(juce::Label::textColourId, juce::Colour(0xFFFFFFFF));
  addAndMakeVisible(bandNameLabel_);

  // Frequency slider (all bands except Low Cut have frequency)
  if (bandType_ != BandType::LowCut || bandIndex_ == 0) {
    freqSlider_ = std::make_unique<juce::Slider>(juce::Slider::RotaryVerticalDrag,
                                                  juce::Slider::TextBoxBelow);
    freqSlider_->setSliderSnapsToMousePosition(false);
    freqSlider_->setScrollWheelEnabled(false);
    freqLabel_.setText("Freq", juce::dontSendNotification);
    freqLabel_.attachToComponent(freqSlider_.get(), false);
    freqLabel_.setJustificationType(juce::Justification::centred);
    freqLabel_.setColour(juce::Label::textColourId, juce::Colour(0xFFCCCCCC));
    addAndMakeVisible(freqSlider_.get());
    addAndMakeVisible(freqLabel_);

    freqAttachment_ = std::make_unique<
        juce::AudioProcessorValueTreeState::SliderAttachment>(apvts, getFreqParamID(),
                                                              *freqSlider_);
  }

  // Gain slider (all bands except Low Cut)
  if (bandType_ != BandType::LowCut) {
    gainSlider_ = std::make_unique<juce::Slider>(juce::Slider::RotaryVerticalDrag,
                                                  juce::Slider::TextBoxBelow);
    gainSlider_->setSliderSnapsToMousePosition(false);
    gainSlider_->setScrollWheelEnabled(false);
    gainLabel_.setText("Gain", juce::dontSendNotification);
    gainLabel_.attachToComponent(gainSlider_.get(), false);
    gainLabel_.setJustificationType(juce::Justification::centred);
    gainLabel_.setColour(juce::Label::textColourId, juce::Colour(0xFFCCCCCC));
    addAndMakeVisible(gainSlider_.get());
    addAndMakeVisible(gainLabel_);

    gainAttachment_ = std::make_unique<
        juce::AudioProcessorValueTreeState::SliderAttachment>(apvts, getGainParamID(),
                                                              *gainSlider_);
  }

  // Q slider (all bands except Low Cut)
  if (bandType_ != BandType::LowCut) {
    qSlider_ = std::make_unique<juce::Slider>(juce::Slider::RotaryVerticalDrag,
                                               juce::Slider::TextBoxBelow);
    qSlider_->setSliderSnapsToMousePosition(false);
    qSlider_->setScrollWheelEnabled(false);
    qLabel_.setText("Q", juce::dontSendNotification);
    qLabel_.attachToComponent(qSlider_.get(), false);
    qLabel_.setJustificationType(juce::Justification::centred);
    qLabel_.setColour(juce::Label::textColourId, juce::Colour(0xFFCCCCCC));
    addAndMakeVisible(qSlider_.get());
    addAndMakeVisible(qLabel_);

    qAttachment_ = std::make_unique<
        juce::AudioProcessorValueTreeState::SliderAttachment>(apvts, getQParamID(),
                                                              *qSlider_);
  }

  // Enable toggle button (all bands)
  enableButton_ = std::make_unique<juce::ToggleButton>();
  enableLabel_.setText("On", juce::dontSendNotification);
  enableLabel_.attachToComponent(enableButton_.get(), false);
  enableLabel_.setJustificationType(juce::Justification::centred);
  enableLabel_.setColour(juce::Label::textColourId, juce::Colour(0xFFCCCCCC));
  addAndMakeVisible(enableButton_.get());
  addAndMakeVisible(enableLabel_);

  enableAttachment_ = std::make_unique<
      juce::AudioProcessorValueTreeState::ButtonAttachment>(apvts, getEnableParamID(),
                                                            *enableButton_);
}

void EQBandControls::paint(juce::Graphics &g) {
  // Background - transparent (parent handles background)
  g.fillAll(juce::Colours::transparentBlack);
}

void EQBandControls::resized() {
  auto bounds = getLocalBounds();
  const int labelHeight = 20;
  const int spacing = 5;

  // Band name at top
  bandNameLabel_.setBounds(bounds.removeFromTop(labelHeight));
  bounds.removeFromTop(spacing);

  // Enable button at bottom (label 15px + button 20px = 35px total)
  const int buttonHeight = 35;
  auto buttonArea = bounds.removeFromBottom(buttonHeight);
  enableLabel_.setBounds(buttonArea.removeFromTop(15));
  enableButton_->setBounds(buttonArea);
  bounds.removeFromBottom(spacing);

  // Sliders in remaining space (equal width)
  if (freqSlider_ && gainSlider_ && qSlider_) {
    // 3 sliders: Freq, Gain, Q
    const int sliderWidth = bounds.getWidth() / 3;
    freqSlider_->setBounds(bounds.removeFromLeft(sliderWidth));
    gainSlider_->setBounds(bounds.removeFromLeft(sliderWidth));
    qSlider_->setBounds(bounds);
  } else if (freqSlider_) {
    // Only frequency (Low Cut)
    freqSlider_->setBounds(bounds);
  }
}

const char *EQBandControls::getFreqParamID() const {
  switch (bandIndex_) {
  case 0:
    return PluginProcessor::PARAM_EQ_BAND_0_FREQ;
  case 1:
    return PluginProcessor::PARAM_EQ_BAND_1_FREQ;
  case 2:
    return PluginProcessor::PARAM_EQ_BAND_2_FREQ;
  case 3:
    return PluginProcessor::PARAM_EQ_BAND_3_FREQ;
  case 4:
    return PluginProcessor::PARAM_EQ_BAND_4_FREQ;
  case 5:
    return PluginProcessor::PARAM_EQ_BAND_5_FREQ;
  default:
    jassertfalse;
    return PluginProcessor::PARAM_EQ_BAND_0_FREQ;
  }
}

const char *EQBandControls::getGainParamID() const {
  switch (bandIndex_) {
  case 1:
    return PluginProcessor::PARAM_EQ_BAND_1_GAIN;
  case 2:
    return PluginProcessor::PARAM_EQ_BAND_2_GAIN;
  case 3:
    return PluginProcessor::PARAM_EQ_BAND_3_GAIN;
  case 4:
    return PluginProcessor::PARAM_EQ_BAND_4_GAIN;
  case 5:
    return PluginProcessor::PARAM_EQ_BAND_5_GAIN;
  default:
    jassertfalse;
    return PluginProcessor::PARAM_EQ_BAND_1_GAIN;
  }
}

const char *EQBandControls::getQParamID() const {
  switch (bandIndex_) {
  case 1:
    return PluginProcessor::PARAM_EQ_BAND_1_Q;
  case 2:
    return PluginProcessor::PARAM_EQ_BAND_2_Q;
  case 3:
    return PluginProcessor::PARAM_EQ_BAND_3_Q;
  case 4:
    return PluginProcessor::PARAM_EQ_BAND_4_Q;
  case 5:
    return PluginProcessor::PARAM_EQ_BAND_5_Q;
  default:
    jassertfalse;
    return PluginProcessor::PARAM_EQ_BAND_1_Q;
  }
}

const char *EQBandControls::getEnableParamID() const {
  switch (bandIndex_) {
  case 0:
    return PluginProcessor::PARAM_EQ_BAND_0_ENABLED;
  case 1:
    return PluginProcessor::PARAM_EQ_BAND_1_ENABLED;
  case 2:
    return PluginProcessor::PARAM_EQ_BAND_2_ENABLED;
  case 3:
    return PluginProcessor::PARAM_EQ_BAND_3_ENABLED;
  case 4:
    return PluginProcessor::PARAM_EQ_BAND_4_ENABLED;
  case 5:
    return PluginProcessor::PARAM_EQ_BAND_5_ENABLED;
  default:
    jassertfalse;
    return PluginProcessor::PARAM_EQ_BAND_0_ENABLED;
  }
}

juce::String EQBandControls::getBandName() const {
  switch (bandIndex_) {
  case 0:
    return "Low Cut";
  case 1:
    return "Low Shelf";
  case 2:
    return "Param 1";
  case 3:
    return "Param 2";
  case 4:
    return "Param 3";
  case 5:
    return "High Shelf";
  default:
    return "Band";
  }
}

} // namespace kelly
