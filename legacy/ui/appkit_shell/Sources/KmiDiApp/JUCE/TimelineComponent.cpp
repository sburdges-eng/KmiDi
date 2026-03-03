#include "TimelineComponent.h"
#include "MLVisualizationLayer.h"

#include <juce_graphics/juce_graphics.h>
#include <juce_events/juce_events.h>
#include <fstream>
#include <sstream>
#include <cmath>

namespace {
class FlatLookAndFeel : public juce::LookAndFeel_V4 {
public:
    FlatLookAndFeel() {
        setColour(juce::ResizableWindow::backgroundColourId, juce::Colours::transparentBlack);
    }
};
} // namespace

TimelineComponent::TimelineComponent() {
    laf_ = std::make_unique<FlatLookAndFeel>();
    setLookAndFeel(laf_.get());
    setOpaque(true);
    startTimerHz(30); // Lightweight playhead animation without heavy redraws

    mlLayer_ = std::make_unique<MLVisualizationLayer>();
    mlLayer_->setVisible(false);
    addAndMakeVisible(mlLayer_.get());
}

void TimelineComponent::setUseDarkMode(bool isDark) {
    isDark_ = isDark;
    if (mlLayer_)
        mlLayer_->setDarkMode(isDark);
    repaint();
}

void TimelineComponent::paint(juce::Graphics& g) {
    auto bounds = getLocalBounds().toFloat();
    const juce::Colour bg = isDark_
        ? juce::Colour(0xFF1A1A1A)
        : juce::Colour(0xFFF2F2F2);
    const juce::Colour grid = isDark_
        ? juce::Colour(0xFF2A2A2A)
        : juce::Colour(0xFFD9D9D9);
    const juce::Colour playhead = juce::Colours::systemBlue;

    g.fillAll(bg);

    // Emotion-driven background tint (extremely subtle, 2-6% opacity)
    updateEmotionTint();
    if (std::abs(cachedValence_) > 0.01f) {
        const juce::Colour tint = getEmotionTintColour(cachedValence_);
        g.setColour(tint);
        g.fillAll();
    }

    // Lightweight grid
    const float beatWidth = 80.0f;
    for (float x = 0.0f; x < bounds.getWidth(); x += beatWidth) {
        g.setColour(grid);
        g.drawLine(x, 0.0f, x, bounds.getHeight(), 1.0f);
    }

    // Playhead
    g.setColour(playhead);
    g.drawLine(playheadX_, 0.0f, playheadX_, bounds.getHeight(), 2.0f);
}

void TimelineComponent::resized() {
    if (mlLayer_)
        mlLayer_->setBounds(getLocalBounds());
}

void TimelineComponent::handleMouse(juce::MouseInputSource::MouseEventType type,
                                    juce::Point<float> pos,
                                    const juce::ModifierKeys& modifiers,
                                    int /*clickCount*/) {
    juce::ignoreUnused(modifiers);
    if (type == juce::MouseInputSource::MouseEventType::down ||
        type == juce::MouseInputSource::MouseEventType::drag) {
        playheadX_ = juce::jlimit(0.0f, (float)getWidth(), pos.x);
        repaint();
    }
}

void TimelineComponent::handleScroll(juce::Point<float> pos, float deltaX, float deltaY, bool precise) {
    juce::ignoreUnused(pos, precise);
    const float factor = 0.1f;
    playheadX_ = juce::jlimit(0.0f, (float)getWidth(), playheadX_ - (deltaX * factor) - (deltaY * factor));
    repaint();
}

void TimelineComponent::handleKeyDown(unsigned short keyCode, NSEventModifierFlags modifiers) {
    juce::ignoreUnused(modifiers);
    // Basic transport-style nudge for demonstration (left/right arrows)
    const float nudge = 5.0f;
    if (keyCode == 123) { // left
        playheadX_ = juce::jmax(0.0f, playheadX_ - nudge);
        repaint();
    } else if (keyCode == 124) { // right
        playheadX_ = juce::jmin((float)getWidth(), playheadX_ + nudge);
        repaint();
    }
}

void TimelineComponent::handleKeyUp(unsigned short keyCode, NSEventModifierFlags modifiers) {
    juce::ignoreUnused(keyCode, modifiers);
}

void TimelineComponent::timerCallback() {
    updatePlayhead();
}

void TimelineComponent::updatePlayhead() {
    // Placeholder animation: drift slowly to the right.
    const float delta = 1.5f;
    playheadX_ += delta;
    if (playheadX_ > (float)getWidth()) {
        playheadX_ = 0.0f;
    }
    repaint();
}

// ML Visualization API
struct KellyMLVisData {
    std::array<float, 128> noteProbabilities {};
    std::array<float, 64>  chordProbabilities {};
    std::array<float, 32>  groove {};
    std::array<float, 16>  dynamics {};
    float valence {0.0f};
    float arousal {0.0f};
    float dominance {0.0f};
    float complexity {0.0f};
};

void TimelineComponent::setMLVisualizationEnabled(bool enabled) {
    mlVisEnabled_ = enabled;
    if (mlLayer_) {
        mlLayer_->setVisible(enabled);
        if (!enabled)
            mlLayer_->clearData();
    }
}

void TimelineComponent::updateMLVisualization(const KellyMLVisData& data) {
    if (!mlVisEnabled_ || !mlLayer_)
        return;
    MLVisualizationLayer::Data d{};
    d.noteProbabilities = data.noteProbabilities;
    d.chordProbabilities = data.chordProbabilities;
    d.groove = data.groove;
    d.dynamics = data.dynamics;
    d.valence = data.valence;
    d.arousal = data.arousal;
    d.dominance = data.dominance;
    d.complexity = data.complexity;
    mlLayer_->setData(d);
}

void TimelineComponent::setVisualizationZoom(float zoom) {
    if (mlLayer_)
        mlLayer_->setZoom(zoom);
}

void TimelineComponent::setVisualizationCategoryEnabled(bool melody, bool harmony, bool groove, bool dynamics, bool emotion) {
    if (!mlLayer_)
        return;
    mlLayer_->setShowMelody(melody);
    mlLayer_->setShowHarmony(harmony);
    mlLayer_->setShowGroove(groove);
    mlLayer_->setShowDynamics(dynamics);
    mlLayer_->setShowEmotion(emotion);
}

void TimelineComponent::setEmotionSnapshotPath(const std::string& path) {
    emotionSnapshotPath_ = path;
    cachedValence_ = 0.0f;
    lastSnapshotCheck_ = 0;
}

void TimelineComponent::updateEmotionTint() {
    if (emotionSnapshotPath_.empty())
        return;

    // Throttle: only check snapshot file every 0.5 seconds
    const auto now = juce::Time::currentTimeMillis() * 1000; // microseconds
    if (now - lastSnapshotCheck_ < snapshotCheckInterval_)
        return;

    lastSnapshotCheck_ = now;

    // Simple JSON parsing: extract valence value
    // This is a minimal parser for "valence": value pattern
    std::ifstream file(emotionSnapshotPath_);
    if (!file.is_open())
        return;

    std::string line;
    while (std::getline(file, line)) {
        // Look for "valence": pattern
        const std::string search = "\"valence\":";
        const auto pos = line.find(search);
        if (pos != std::string::npos) {
            // Extract number after colon
            const auto start = pos + search.length();
            const auto end = line.find_first_of(",}", start);
            if (end != std::string::npos) {
                const std::string valStr = line.substr(start, end - start);
                try {
                    cachedValence_ = std::stof(valStr);
                } catch (...) {
                    // Ignore parse errors
                }
            }
            break;
        }
    }
}

juce::Colour TimelineComponent::getEmotionTintColour(float valence) const {
    // Map valence (-1.0 to 1.0) to warm/cool hue bias
    // Negative valence → cool (blue/cyan), positive → warm (orange/red)
    // Extremely subtle: 2-6% opacity max

    const float opacity = std::min(0.06f, std::abs(valence) * 0.04f + 0.02f);

    if (valence < 0.0f) {
        // Cool tones (blue/cyan)
        const float coolness = std::abs(valence);
        const juce::Colour cool = juce::Colour::fromFloatRGBA(0.3f, 0.5f, 0.8f, opacity);
        return cool;
    } else {
        // Warm tones (orange/red)
        const float warmth = valence;
        const juce::Colour warm = juce::Colour::fromFloatRGBA(0.9f, 0.6f, 0.3f, opacity);
        return warm;
    }
}
