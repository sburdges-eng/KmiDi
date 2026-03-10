#include "plugin/WaveformView.h"

namespace kelly {

void WaveformView::setAudioBuffer(const float* data, size_t numSamples) {
    buffer_.clear();
    if (data && numSamples > 0) {
        buffer_.reserve(numSamples);
        for (size_t i = 0; i < numSamples; ++i)
            buffer_.push_back(data[i]);
    }
    repaint();
}

void WaveformView::setAudioBuffer(const std::vector<float>& buf) {
    if (buf.empty())
        setAudioBuffer(nullptr, 0);
    else
        setAudioBuffer(buf.data(), buf.size());
}

void WaveformView::paint(juce::Graphics& g) {
    auto b = getLocalBounds().toFloat().reduced(2.0f);

    g.setColour(juce::Colour(0xFFE8E8E8));
    g.fillRoundedRectangle(b, 4.0f);

    g.setColour(juce::Colours::darkgrey);
    g.drawRoundedRectangle(b, 4.0f, 1.0f);

    if (buffer_.empty()) {
        g.setColour(juce::Colours::grey);
        g.setFont(juce::Font(12.0f));
        g.drawText("Generate to see waveform", b, juce::Justification::centred);
        return;
    }

    g.setColour(juce::Colour(0xFF4A90D9));
    const float w = b.getWidth();
    const float h = b.getHeight();
    const float midY = b.getY() + h * 0.5f;
    const size_t n = buffer_.size();
    const int displayPoints = std::max(1, std::min(static_cast<int>(w), 800));

    juce::Path path;
    path.startNewSubPath(b.getX(), midY);

    for (int i = 0; i <= displayPoints; ++i) {
        const size_t idx = (n > 0) ? (i * (n - 1)) / displayPoints : 0;
        const float x = b.getX() + (static_cast<float>(i) / static_cast<float>(displayPoints)) * w;
        const float y = midY - buffer_[idx] * h * 0.4f;
        path.lineTo(x, y);
    }
    path.lineTo(b.getRight(), midY);
    path.closeSubPath();
    g.fillPath(path);
}

} // namespace kelly
