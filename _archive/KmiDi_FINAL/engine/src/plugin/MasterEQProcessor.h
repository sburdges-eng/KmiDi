#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <array>

namespace kelly {

/**
 * MasterEQProcessor - DSP processor for Master EQ
 *
 * Handles parameter smoothing and audio processing for the 6-band EQ.
 * Currently stubbed for audio processing - filters will be implemented later.
 *
 * Thread Safety:
 * - prepareToPlay() and processBlock() are called from audio thread
 * - updateParameters() can be called from message thread
 * - All smoothed values are lock-free and thread-safe
 */
class MasterEQProcessor {
public:
  MasterEQProcessor();
  ~MasterEQProcessor() = default;

  /**
   * Prepare for playback - initialize smoothed values and filter state
   */
  void prepareToPlay(double sampleRate, int samplesPerBlock, int numChannels);

  /**
   * Release resources
   */
  void releaseResources();

  /**
   * Process audio block through EQ
   * Audio processing is currently stubbed - applies bypass state only
   */
  void processBlock(juce::AudioBuffer<float> &buffer);

  /**
   * Update parameter targets from APVTS (called from parameterChanged callback)
   * This sets the target values for smoothed parameters
   */
  void updateParameters(juce::AudioProcessorValueTreeState &apvts);

  /**
   * Get current smoothed parameter values (for UI display)
   * These are the actual values being used in audio processing
   */
  struct EQBandState {
    float freq = 1000.0f;
    float gain = 0.0f;
    float q = 1.0f;
    bool enabled = true;
  };

  struct EQState {
    bool bypass = false;
    EQBandState bands[6];
  };

  EQState getCurrentState() const;

private:
  // Smoothed values for each band parameter
  // Time constants: Gain=30ms, Freq=60ms, Q=120ms
  struct BandSmoothing {
    juce::SmoothedValue<float> freq;
    juce::SmoothedValue<float> gain;
    juce::SmoothedValue<float> q;
    bool enabled = true;
  };

  std::array<BandSmoothing, 6> bandSmoothing_;
  juce::SmoothedValue<float> bypassSmoothing_; // 5ms crossfade

  double currentSampleRate_ = 44100.0;
  int currentNumChannels_ = 2;

  std::array<juce::dsp::IIR::Coefficients<float>::Ptr, 6> bandCoefficients_;
  std::vector<std::array<juce::dsp::IIR::Filter<float>, 6>> bandFilters_;
  juce::AudioBuffer<float> dryBuffer_;

  // Current parameter targets (atomic-safe reads)
  bool eqBypass_ = false;
  std::array<bool, 6> bandEnabled_;

  /**
   * Calculate smoothing coefficient from time constant
   * α = 1 - exp(-1 / (T * Fs))
   */
  static float calculateSmoothingCoeff(double timeConstantSeconds, double sampleRate);
};

} // namespace kelly
