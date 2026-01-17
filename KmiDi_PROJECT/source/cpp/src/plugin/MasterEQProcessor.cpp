#include "plugin/MasterEQProcessor.h"
#include "plugin/PluginProcessor.h"

namespace kelly {

MasterEQProcessor::MasterEQProcessor() {
  // Initialize all smoothed values to their defaults
  for (auto &band : bandSmoothing_) {
    band.freq.reset(1000.0f);
    band.gain.reset(0.0f);
    band.q.reset(1.0f);
    band.enabled = true;
  }
  bypassSmoothing_.reset(0.0f);
}

void MasterEQProcessor::prepareToPlay(double sampleRate, int samplesPerBlock,
                                      int numChannels) {
  currentSampleRate_ = sampleRate;
  currentNumChannels_ = numChannels;

  // Set smoothing coefficients based on sample rate
  const float gainCoeff = calculateSmoothingCoeff(0.030, sampleRate);    // 30ms
  const float freqCoeff = calculateSmoothingCoeff(0.060, sampleRate);    // 60ms
  const float qCoeff = calculateSmoothingCoeff(0.120, sampleRate);       // 120ms
  const float bypassCoeff = calculateSmoothingCoeff(0.005, sampleRate);  // 5ms

  for (auto &band : bandSmoothing_) {
    band.freq.reset(sampleRate, 0.060f); // 60ms
    band.freq.setCurrentAndTargetValue(1000.0f);

    band.gain.reset(sampleRate, 0.030f); // 30ms
    band.gain.setCurrentAndTargetValue(0.0f);

    band.q.reset(sampleRate, 0.120f); // 120ms
    band.q.setCurrentAndTargetValue(1.0f);
  }

  bypassSmoothing_.reset(sampleRate, 0.005f); // 5ms
  bypassSmoothing_.setCurrentAndTargetValue(0.0f);

  // TODO: Initialize filter state when biquad filters are implemented
}

void MasterEQProcessor::releaseResources() {
  // TODO: Clean up filter state when biquad filters are implemented
}

void MasterEQProcessor::processBlock(juce::AudioBuffer<float> &buffer) {
  const int numSamples = buffer.getNumSamples();
  const int numChannels = buffer.getNumChannels();

  // Update smoothed values for this block
  // Note: For per-sample smoothing, call getNextValue() inside the sample loop
  // For block-based smoothing, we can skip smoothing in the stub
  // When real filters are implemented, we'll use per-sample smoothing

  // Check bypass state (read atomic parameter)
  // In stub, just pass through audio unchanged
  // TODO: When filters are implemented, apply EQ processing here

  // Stub: Currently just pass audio through
  // Real implementation will:
  // 1. Read smoothed parameter values per sample or per block
  // 2. Apply biquad filters for each enabled band
  // 3. Apply bypass crossfade if bypass is enabled

  juce::ignoreUnused(numSamples, numChannels);
}

void MasterEQProcessor::updateParameters(juce::AudioProcessorValueTreeState &apvts) {
  // Update EQ bypass target
  auto *bypassParam = apvts.getRawParameterValue(PluginProcessor::PARAM_EQ_BYPASS);
  if (bypassParam) {
    eqBypass_ = (*bypassParam > 0.5f);
    bypassSmoothing_.setTargetValue(eqBypass_ ? 1.0f : 0.0f);
  }

  // Update band 0 (Low Cut) - frequency only
  auto *band0Freq = apvts.getRawParameterValue(PluginProcessor::PARAM_EQ_BAND_0_FREQ);
  auto *band0Enabled = apvts.getRawParameterValue(PluginProcessor::PARAM_EQ_BAND_0_ENABLED);
  if (band0Freq) {
    bandSmoothing_[0].freq.setTargetValue(*band0Freq);
  }
  if (band0Enabled) {
    bandEnabled_[0] = (*band0Enabled > 0.5f);
    bandSmoothing_[0].enabled = bandEnabled_[0];
  }

  // Update band 1 (Low Shelf)
  auto *band1Freq = apvts.getRawParameterValue(PluginProcessor::PARAM_EQ_BAND_1_FREQ);
  auto *band1Gain = apvts.getRawParameterValue(PluginProcessor::PARAM_EQ_BAND_1_GAIN);
  auto *band1Q = apvts.getRawParameterValue(PluginProcessor::PARAM_EQ_BAND_1_Q);
  auto *band1Enabled = apvts.getRawParameterValue(PluginProcessor::PARAM_EQ_BAND_1_ENABLED);
  if (band1Freq) {
    bandSmoothing_[1].freq.setTargetValue(*band1Freq);
  }
  if (band1Gain) {
    bandSmoothing_[1].gain.setTargetValue(*band1Gain);
  }
  if (band1Q) {
    bandSmoothing_[1].q.setTargetValue(*band1Q);
  }
  if (band1Enabled) {
    bandEnabled_[1] = (*band1Enabled > 0.5f);
    bandSmoothing_[1].enabled = bandEnabled_[1];
  }

  // Update band 2 (Parametric)
  auto *band2Freq = apvts.getRawParameterValue(PluginProcessor::PARAM_EQ_BAND_2_FREQ);
  auto *band2Gain = apvts.getRawParameterValue(PluginProcessor::PARAM_EQ_BAND_2_GAIN);
  auto *band2Q = apvts.getRawParameterValue(PluginProcessor::PARAM_EQ_BAND_2_Q);
  auto *band2Enabled = apvts.getRawParameterValue(PluginProcessor::PARAM_EQ_BAND_2_ENABLED);
  if (band2Freq) {
    bandSmoothing_[2].freq.setTargetValue(*band2Freq);
  }
  if (band2Gain) {
    bandSmoothing_[2].gain.setTargetValue(*band2Gain);
  }
  if (band2Q) {
    bandSmoothing_[2].q.setTargetValue(*band2Q);
  }
  if (band2Enabled) {
    bandEnabled_[2] = (*band2Enabled > 0.5f);
    bandSmoothing_[2].enabled = bandEnabled_[2];
  }

  // Update band 3 (Parametric)
  auto *band3Freq = apvts.getRawParameterValue(PluginProcessor::PARAM_EQ_BAND_3_FREQ);
  auto *band3Gain = apvts.getRawParameterValue(PluginProcessor::PARAM_EQ_BAND_3_GAIN);
  auto *band3Q = apvts.getRawParameterValue(PluginProcessor::PARAM_EQ_BAND_3_Q);
  auto *band3Enabled = apvts.getRawParameterValue(PluginProcessor::PARAM_EQ_BAND_3_ENABLED);
  if (band3Freq) {
    bandSmoothing_[3].freq.setTargetValue(*band3Freq);
  }
  if (band3Gain) {
    bandSmoothing_[3].gain.setTargetValue(*band3Gain);
  }
  if (band3Q) {
    bandSmoothing_[3].q.setTargetValue(*band3Q);
  }
  if (band3Enabled) {
    bandEnabled_[3] = (*band3Enabled > 0.5f);
    bandSmoothing_[3].enabled = bandEnabled_[3];
  }

  // Update band 4 (Parametric)
  auto *band4Freq = apvts.getRawParameterValue(PluginProcessor::PARAM_EQ_BAND_4_FREQ);
  auto *band4Gain = apvts.getRawParameterValue(PluginProcessor::PARAM_EQ_BAND_4_GAIN);
  auto *band4Q = apvts.getRawParameterValue(PluginProcessor::PARAM_EQ_BAND_4_Q);
  auto *band4Enabled = apvts.getRawParameterValue(PluginProcessor::PARAM_EQ_BAND_4_ENABLED);
  if (band4Freq) {
    bandSmoothing_[4].freq.setTargetValue(*band4Freq);
  }
  if (band4Gain) {
    bandSmoothing_[4].gain.setTargetValue(*band4Gain);
  }
  if (band4Q) {
    bandSmoothing_[4].q.setTargetValue(*band4Q);
  }
  if (band4Enabled) {
    bandEnabled_[4] = (*band4Enabled > 0.5f);
    bandSmoothing_[4].enabled = bandEnabled_[4];
  }

  // Update band 5 (High Shelf)
  auto *band5Freq = apvts.getRawParameterValue(PluginProcessor::PARAM_EQ_BAND_5_FREQ);
  auto *band5Gain = apvts.getRawParameterValue(PluginProcessor::PARAM_EQ_BAND_5_GAIN);
  auto *band5Q = apvts.getRawParameterValue(PluginProcessor::PARAM_EQ_BAND_5_Q);
  auto *band5Enabled = apvts.getRawParameterValue(PluginProcessor::PARAM_EQ_BAND_5_ENABLED);
  if (band5Freq) {
    bandSmoothing_[5].freq.setTargetValue(*band5Freq);
  }
  if (band5Gain) {
    bandSmoothing_[5].gain.setTargetValue(*band5Gain);
  }
  if (band5Q) {
    bandSmoothing_[5].q.setTargetValue(*band5Q);
  }
  if (band5Enabled) {
    bandEnabled_[5] = (*band5Enabled > 0.5f);
    bandSmoothing_[5].enabled = bandEnabled_[5];
  }
}

MasterEQProcessor::EQState MasterEQProcessor::getCurrentState() const {
  EQState state;
  state.bypass = eqBypass_;

  for (size_t i = 0; i < 6; ++i) {
    state.bands[i].freq = bandSmoothing_[i].freq.getCurrentValue();
    state.bands[i].gain = bandSmoothing_[i].gain.getCurrentValue();
    state.bands[i].q = bandSmoothing_[i].q.getCurrentValue();
    state.bands[i].enabled = bandSmoothing_[i].enabled;
  }

  return state;
}

float MasterEQProcessor::calculateSmoothingCoeff(double timeConstantSeconds,
                                                  double sampleRate) {
  if (timeConstantSeconds <= 0.0 || sampleRate <= 0.0) {
    return 1.0f; // No smoothing
  }
  // α = 1 - exp(-1 / (T * Fs))
  return static_cast<float>(1.0 - std::exp(-1.0 / (timeConstantSeconds * sampleRate)));
}

} // namespace kelly
