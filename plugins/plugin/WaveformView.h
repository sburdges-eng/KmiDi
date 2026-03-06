#pragma once

/**
 * D2 — Read-only waveform display.
 * Consumes PCM only. No intent, generation, playback, or suggestions.
 * Display-only; no feedback loop.
 */

#include <juce_gui_basics/juce_gui_basics.h>
#include <vector>

namespace kelly {

class WaveformView : public juce::Component {
public:
    WaveformView() = default;

    /** Set PCM buffer to display. Empty = show placeholder. */
    void setAudioBuffer(const float* data, size_t numSamples);

    /** Set from vector. */
    void setAudioBuffer(const std::vector<float>& buffer);

    void paint(juce::Graphics& g) override;

private:
    std::vector<float> buffer_;
};

} // namespace kelly
