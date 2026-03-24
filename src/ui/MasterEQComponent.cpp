#include "ui/MasterEQComponent.h"
#include "plugin/PluginProcessor.h"
#include <cmath>

namespace kelly {

MasterEQComponent::MasterEQComponent(juce::AudioProcessorValueTreeState &apvts,
                                     MasterEQProcessor &eqProcessor)
    : apvts_(apvts), eqProcessor_(eqProcessor) {

  // Create curve view
  curveView_ = std::make_unique<EQCurveView>();
  addAndMakeVisible(*curveView_);

  // Create band controls (6 bands)
  bandControls_.reserve(6);
  bandControls_.push_back(
      std::make_unique<EQBandControls>(0, EQBandControls::BandType::LowCut, apvts_));
  bandControls_.push_back(
      std::make_unique<EQBandControls>(1, EQBandControls::BandType::LowShelf, apvts_));
  bandControls_.push_back(
      std::make_unique<EQBandControls>(2, EQBandControls::BandType::Parametric, apvts_));
  bandControls_.push_back(
      std::make_unique<EQBandControls>(3, EQBandControls::BandType::Parametric, apvts_));
  bandControls_.push_back(
      std::make_unique<EQBandControls>(4, EQBandControls::BandType::Parametric, apvts_));
  bandControls_.push_back(
      std::make_unique<EQBandControls>(5, EQBandControls::BandType::HighShelf, apvts_));

  for (auto &band : bandControls_) {
    addAndMakeVisible(*band);
  }

  // AI Controls
  aiControlsLabel_.setText("AI EQ Assist", juce::dontSendNotification);
  aiControlsLabel_.setJustificationType(juce::Justification::centred);
  aiControlsLabel_.setColour(juce::Label::textColourId, juce::Colour(0xFFFFFFFF));
  addAndMakeVisible(aiControlsLabel_);

  aiEnabledButton_ = std::make_unique<juce::ToggleButton>();
  aiEnabledButton_->setButtonText("Enabled");
  addAndMakeVisible(*aiEnabledButton_);
  aiEnabledAttachment_ = std::make_unique<
      juce::AudioProcessorValueTreeState::ButtonAttachment>(
      apvts_, PluginProcessor::PARAM_AI_EQ_ENABLED, *aiEnabledButton_);

  aiIntensityLabel_.setText("Intensity", juce::dontSendNotification);
  aiIntensityLabel_.setJustificationType(juce::Justification::centred);
  aiIntensityLabel_.setColour(juce::Label::textColourId, juce::Colour(0xFFCCCCCC));
  addAndMakeVisible(aiIntensityLabel_);

  aiIntensitySlider_ = std::make_unique<juce::Slider>(juce::Slider::LinearHorizontal,
                                                       juce::Slider::TextBoxRight);
  aiIntensitySlider_->setRange(0.0, 1.0, 0.01);
  addAndMakeVisible(*aiIntensitySlider_);
  aiIntensityAttachment_ = std::make_unique<
      juce::AudioProcessorValueTreeState::SliderAttachment>(
      apvts_, PluginProcessor::PARAM_AI_EQ_INTENSITY, *aiIntensitySlider_);

  lockUserBandsButton_ = std::make_unique<juce::ToggleButton>();
  lockUserBandsButton_->setButtonText("Lock User Bands");
  addAndMakeVisible(*lockUserBandsButton_);
  lockUserBandsAttachment_ = std::make_unique<
      juce::AudioProcessorValueTreeState::ButtonAttachment>(
      apvts_, PluginProcessor::PARAM_AI_EQ_LOCK_USER_BANDS, *lockUserBandsButton_);

  applySuggestedButton_ = std::make_unique<juce::TextButton>("Apply Suggested Curve");
  applySuggestedButton_->onClick = [this]() { applySuggestedCurve(); };
  addAndMakeVisible(*applySuggestedButton_);

  // Listen to parameter changes
  apvts_.addParameterListener(PluginProcessor::PARAM_AI_EQ_ENABLED, this);
  apvts_.addParameterListener(PluginProcessor::PARAM_AI_EQ_INTENSITY, this);
  apvts_.addParameterListener(PluginProcessor::PARAM_AI_EQ_LOCK_USER_BANDS, this);

  // Listen to all EQ band parameters
  apvts_.addParameterListener(PluginProcessor::PARAM_EQ_BAND_0_FREQ, this);
  apvts_.addParameterListener(PluginProcessor::PARAM_EQ_BAND_0_ENABLED, this);
  apvts_.addParameterListener(PluginProcessor::PARAM_EQ_BAND_1_FREQ, this);
  apvts_.addParameterListener(PluginProcessor::PARAM_EQ_BAND_1_GAIN, this);
  apvts_.addParameterListener(PluginProcessor::PARAM_EQ_BAND_1_Q, this);
  apvts_.addParameterListener(PluginProcessor::PARAM_EQ_BAND_1_ENABLED, this);
  apvts_.addParameterListener(PluginProcessor::PARAM_EQ_BAND_2_FREQ, this);
  apvts_.addParameterListener(PluginProcessor::PARAM_EQ_BAND_2_GAIN, this);
  apvts_.addParameterListener(PluginProcessor::PARAM_EQ_BAND_2_Q, this);
  apvts_.addParameterListener(PluginProcessor::PARAM_EQ_BAND_2_ENABLED, this);
  apvts_.addParameterListener(PluginProcessor::PARAM_EQ_BAND_3_FREQ, this);
  apvts_.addParameterListener(PluginProcessor::PARAM_EQ_BAND_3_GAIN, this);
  apvts_.addParameterListener(PluginProcessor::PARAM_EQ_BAND_3_Q, this);
  apvts_.addParameterListener(PluginProcessor::PARAM_EQ_BAND_3_ENABLED, this);
  apvts_.addParameterListener(PluginProcessor::PARAM_EQ_BAND_4_FREQ, this);
  apvts_.addParameterListener(PluginProcessor::PARAM_EQ_BAND_4_GAIN, this);
  apvts_.addParameterListener(PluginProcessor::PARAM_EQ_BAND_4_Q, this);
  apvts_.addParameterListener(PluginProcessor::PARAM_EQ_BAND_4_ENABLED, this);
  apvts_.addParameterListener(PluginProcessor::PARAM_EQ_BAND_5_FREQ, this);
  apvts_.addParameterListener(PluginProcessor::PARAM_EQ_BAND_5_GAIN, this);
  apvts_.addParameterListener(PluginProcessor::PARAM_EQ_BAND_5_Q, this);
  apvts_.addParameterListener(PluginProcessor::PARAM_EQ_BAND_5_ENABLED, this);

  // Start timer for curve updates (30Hz = 33ms)
  startTimer(33);

  // Initialize curves
  updateUserCurve();
}

MasterEQComponent::~MasterEQComponent() {
  // Stop timer before destruction to prevent callbacks on destroyed object
  stopTimer();

  // Remove all parameter listeners to prevent callbacks on destroyed object
  apvts_.removeParameterListener(PluginProcessor::PARAM_AI_EQ_ENABLED, this);
  apvts_.removeParameterListener(PluginProcessor::PARAM_AI_EQ_INTENSITY, this);
  apvts_.removeParameterListener(PluginProcessor::PARAM_AI_EQ_LOCK_USER_BANDS, this);

  // Remove all EQ band parameter listeners
  apvts_.removeParameterListener(PluginProcessor::PARAM_EQ_BAND_0_FREQ, this);
  apvts_.removeParameterListener(PluginProcessor::PARAM_EQ_BAND_0_ENABLED, this);
  apvts_.removeParameterListener(PluginProcessor::PARAM_EQ_BAND_1_FREQ, this);
  apvts_.removeParameterListener(PluginProcessor::PARAM_EQ_BAND_1_GAIN, this);
  apvts_.removeParameterListener(PluginProcessor::PARAM_EQ_BAND_1_Q, this);
  apvts_.removeParameterListener(PluginProcessor::PARAM_EQ_BAND_1_ENABLED, this);
  apvts_.removeParameterListener(PluginProcessor::PARAM_EQ_BAND_2_FREQ, this);
  apvts_.removeParameterListener(PluginProcessor::PARAM_EQ_BAND_2_GAIN, this);
  apvts_.removeParameterListener(PluginProcessor::PARAM_EQ_BAND_2_Q, this);
  apvts_.removeParameterListener(PluginProcessor::PARAM_EQ_BAND_2_ENABLED, this);
  apvts_.removeParameterListener(PluginProcessor::PARAM_EQ_BAND_3_FREQ, this);
  apvts_.removeParameterListener(PluginProcessor::PARAM_EQ_BAND_3_GAIN, this);
  apvts_.removeParameterListener(PluginProcessor::PARAM_EQ_BAND_3_Q, this);
  apvts_.removeParameterListener(PluginProcessor::PARAM_EQ_BAND_3_ENABLED, this);
  apvts_.removeParameterListener(PluginProcessor::PARAM_EQ_BAND_4_FREQ, this);
  apvts_.removeParameterListener(PluginProcessor::PARAM_EQ_BAND_4_GAIN, this);
  apvts_.removeParameterListener(PluginProcessor::PARAM_EQ_BAND_4_Q, this);
  apvts_.removeParameterListener(PluginProcessor::PARAM_EQ_BAND_4_ENABLED, this);
  apvts_.removeParameterListener(PluginProcessor::PARAM_EQ_BAND_5_FREQ, this);
  apvts_.removeParameterListener(PluginProcessor::PARAM_EQ_BAND_5_GAIN, this);
  apvts_.removeParameterListener(PluginProcessor::PARAM_EQ_BAND_5_Q, this);
  apvts_.removeParameterListener(PluginProcessor::PARAM_EQ_BAND_5_ENABLED, this);
}

void MasterEQComponent::paint(juce::Graphics &g) {
  // Background
  g.fillAll(juce::Colour(0xFF1A1A1A));
}

void MasterEQComponent::resized() {
  auto bounds = getLocalBounds();
  const int aiControlsHeight = 80;
  const int bandControlsHeight = 120;
  const int spacing = 5;

  // AI controls at top
  auto aiArea = bounds.removeFromTop(aiControlsHeight);
  aiControlsLabel_.setBounds(aiArea.removeFromTop(20));
  aiArea.removeFromTop(spacing);

  const int controlWidth = aiArea.getWidth() / 4;
  aiEnabledButton_->setBounds(aiArea.removeFromLeft(controlWidth));
  aiIntensityLabel_.setBounds(aiArea.removeFromLeft(controlWidth).removeFromTop(15));
  aiIntensitySlider_->setBounds(aiArea.removeFromLeft(controlWidth));
  lockUserBandsButton_->setBounds(aiArea.removeFromLeft(controlWidth));
  applySuggestedButton_->setBounds(aiArea);

  bounds.removeFromTop(spacing);

  // Curve view in middle
  const int curveHeight = 200;
  curveView_->setBounds(bounds.removeFromTop(curveHeight));
  bounds.removeFromTop(spacing);

  // Band controls at bottom
  const int bandWidth = bounds.getWidth() / 6;
  for (size_t i = 0; i < bandControls_.size(); ++i) {
    bandControls_[i]->setBounds(bounds.removeFromLeft(bandWidth));
  }
}

void MasterEQComponent::parameterChanged(const juce::String &parameterID,
                                         float newValue) {
  juce::ignoreUnused(newValue);

  // Update AI state
  if (parameterID == PluginProcessor::PARAM_AI_EQ_ENABLED) {
    auto *param = apvts_.getRawParameterValue(parameterID);
    if (param) {
      aiEnabled_ = (*param > 0.5f);
      curveView_->setAICurveVisible(aiEnabled_);
      if (!aiEnabled_) {
        aiCurveValid_ = false; // Invalidate AI curve when disabled
      }
    }
  } else if (parameterID == PluginProcessor::PARAM_AI_EQ_INTENSITY) {
    auto *param = apvts_.getRawParameterValue(parameterID);
    if (param) {
      aiIntensity_ = *param;
      curveView_->setAICurveIntensity(aiIntensity_);
      aiCurveValid_ = false; // Invalidate to recompute with new intensity
    }
  } else if (parameterID == PluginProcessor::PARAM_AI_EQ_LOCK_USER_BANDS) {
    auto *param = apvts_.getRawParameterValue(parameterID);
    if (param) {
      lockUserBands_ = (*param > 0.5f);
      aiCurveValid_ = false; // Invalidate to recompute with new lock setting
    }
  }

  // EQ band parameter changed - invalidate user curve
  if (parameterID.startsWith("eq_band_")) {
    userCurveValid_ = false;
    // If AI is enabled, also invalidate AI curve
    if (aiEnabled_) {
      aiCurveValid_ = false;
    }
  }
}

void MasterEQComponent::timerCallback() {
  // Update user curve if invalid
  if (!userCurveValid_) {
    updateUserCurve();
  }

  // Update AI curve if enabled and invalid
  if (aiEnabled_ && !aiCurveValid_) {
    updateAICurve();
  }
}

void MasterEQComponent::updateEmotionState(
    const kelly::ml::EmotionState &emotionState) {
  currentEmotionState_ = emotionState;
  aiCurveValid_ = false; // Invalidate AI curve to recompute with new emotion
}

void MasterEQComponent::applySuggestedCurve() {
  // Per spec 07: AI suggestions clamped to +/-1.5 dB max adjustment per band.
  // User curve has absolute priority -- AI only suggests, never overrides.
  // ai_eq_intensity (0-1) scales how much of the suggestion is applied.
  // If ai_eq_lock_user_bands is true, bands the user manually adjusted are skipped.
  static constexpr float kMaxAISuggestionDb = 1.5f;

  if (!aiEnabled_) {
    return;
  }

  // Read current AI settings
  auto *intensityParam =
      apvts_.getRawParameterValue(PluginProcessor::PARAM_AI_EQ_INTENSITY);
  const float aiIntensity =
      intensityParam ? static_cast<float>(*intensityParam) : 0.5f;

  auto *lockParam =
      apvts_.getRawParameterValue(PluginProcessor::PARAM_AI_EQ_LOCK_USER_BANDS);
  const bool lockBands = lockParam ? (*lockParam > 0.5f) : false;

  // Get the base AI curve (raw suggestion from emotion state)
  auto baseCurve = aiEngine_.generateBaseAICurve(currentEmotionState_);

  // Get current EQ state to read user gains
  auto eqState = eqProcessor_.getCurrentState();

  // Per-band gain parameter IDs (bands 0 has no gain -- low cut only)
  static const char *bandGainParams[6] = {
      nullptr, // Band 0: Low Cut -- no gain parameter
      PluginProcessor::PARAM_EQ_BAND_1_GAIN,
      PluginProcessor::PARAM_EQ_BAND_2_GAIN,
      PluginProcessor::PARAM_EQ_BAND_3_GAIN,
      PluginProcessor::PARAM_EQ_BAND_4_GAIN,
      PluginProcessor::PARAM_EQ_BAND_5_GAIN,
  };

  // Map the suggested curve to per-band gain adjustments
  const auto &freqGrid = AIEQSuggestionEngine::getFrequencyGrid();

  for (int band = 0; band < 6; ++band) {
    // Band 0 (Low Cut) has no gain parameter -- skip
    if (bandGainParams[band] == nullptr) {
      continue;
    }

    // If lock_user_bands is true, skip bands the user has manually adjusted
    // (gain != 0.0 indicates user has touched the band)
    if (lockBands) {
      float userGain = eqState.bands[band].gain;
      if (std::abs(userGain) > 0.01f) {
        continue; // User has adjusted this band -- skip per spec
      }
    }

    // Find the curve point nearest to this band's center frequency
    float bandFreq = eqState.bands[band].freq;
    int nearestIdx = 0;
    float minDist = std::abs(freqGrid[0] - bandFreq);
    for (int i = 1; i < AIEQSuggestionEngine::CURVE_POINTS; ++i) {
      float dist = std::abs(freqGrid[i] - bandFreq);
      if (dist < minDist) {
        minDist = dist;
        nearestIdx = i;
      }
    }

    // Get AI suggested gain at band center, clamp to +/-1.5 dB
    float aiGain = baseCurve[static_cast<size_t>(nearestIdx)];
    aiGain = juce::jlimit(-kMaxAISuggestionDb, kMaxAISuggestionDb, aiGain);

    // Scale by AI intensity
    float suggestedGain = aiGain * aiIntensity;

    // Blend: user gain + scaled AI suggestion
    float userGain = eqState.bands[band].gain;
    float newGain = userGain + suggestedGain;

    // Apply to APVTS parameter (clamped to parameter range by JUCE)
    auto *param = apvts_.getParameter(bandGainParams[band]);
    if (param != nullptr) {
      float normValue = param->convertTo0to1(newGain);
      param->setValueNotifyingHost(normValue);
    }
  }
}

void MasterEQComponent::updateUserCurve() {
  // Get current EQ state
  auto eqState = eqProcessor_.getCurrentState();
  const auto &freqGrid = AIEQSuggestionEngine::getFrequencyGrid();

  // Compute curve response
  for (int i = 0; i < AIEQSuggestionEngine::CURVE_POINTS; ++i) {
    userCurve_[i] = computeEQResponse(freqGrid[i], eqState);
  }

  // Update curve view
  curveView_->updateUserCurve(userCurve_);
  userCurveValid_ = true;
}

void MasterEQComponent::updateAICurve() {
  // Get current EQ state
  auto eqState = eqProcessor_.getCurrentState();

  // Get AI intensity with null check
  // getRawParameterValue returns std::atomic<float>*, need explicit conversion
  auto *intensityParam = apvts_.getRawParameterValue(PluginProcessor::PARAM_AI_EQ_INTENSITY);
  float aiIntensity = intensityParam ? static_cast<float>(*intensityParam) : 0.5f;

  // Get lock bands setting with null check
  auto *lockParam = apvts_.getRawParameterValue(PluginProcessor::PARAM_AI_EQ_LOCK_USER_BANDS);
  bool lockBands = lockParam ? (*lockParam > 0.5f) : false;

  // Ensure user curve is valid
  if (!userCurveValid_) {
    updateUserCurve();
  }

  // Generate AI suggestion
  aiSuggestedCurve_ = aiEngine_.generateSuggestion(currentEmotionState_, userCurve_,
                                                    eqState, aiIntensity, lockBands);

  // Update curve view
  curveView_->updateAICurve(aiSuggestedCurve_);
  aiCurveValid_ = true;
}

float MasterEQComponent::computeEQResponse(
    float freq, const MasterEQProcessor::EQState &state) const {
  float response = 0.0f;

  // TODO: Compute actual EQ response from band parameters
  // For now, this is a simplified implementation
  // Real implementation would compute biquad filter response

  for (int i = 0; i < 6; ++i) {
    if (!state.bands[i].enabled) {
      continue;
    }

    float bandFreq = state.bands[i].freq;
    float bandGain = state.bands[i].gain;
    float bandQ = state.bands[i].q;

    // Simplified frequency response (Bell curve approximation)
    if (i == 0) {
      // Low Cut: -12dB/octave below frequency
      if (freq < bandFreq) {
        float ratio = freq / bandFreq;
        response -= 12.0f * std::log2(1.0f / ratio); // -12dB per octave
      }
    } else if (i == 1) {
      // Low Shelf: approximate shelf response
      if (freq < bandFreq) {
        float ratio = freq / bandFreq;
        response += bandGain * (1.0f - ratio);
      }
    } else if (i == 5) {
      // High Shelf: approximate shelf response
      if (freq > bandFreq) {
        float ratio = freq / bandFreq;
        response += bandGain * (1.0f - 1.0f / ratio);
      }
    } else {
      // Parametric: bell curve
      float ratio = freq / bandFreq;
      float bandwidth = 1.0f / bandQ;
      float normalizedFreq = std::log2(ratio) / bandwidth;
      float bell = bandGain / (1.0f + normalizedFreq * normalizedFreq);
      response += bell;
    }
  }

  return response;
}

} // namespace kelly
