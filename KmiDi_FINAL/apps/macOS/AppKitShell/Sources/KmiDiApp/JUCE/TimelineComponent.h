#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

class MLVisualizationLayer;

// Minimal JUCE timeline component used for drawing timeline and handling localized input.
class TimelineComponent : public juce::Component, private juce::Timer {
public:
    TimelineComponent();
    ~TimelineComponent() override = default;

    void paint(juce::Graphics& g) override;
    void resized() override;

    // Appearance
    void setUseDarkMode(bool isDark);

    // ML visualization hooks (optional, non-intrusive).
    void setMLVisualizationEnabled(bool enabled);
    void updateMLVisualization(const struct KellyMLVisData& data);
    void setVisualizationZoom(float zoom);
    void setVisualizationCategoryEnabled(bool melody, bool harmony, bool groove, bool dynamics, bool emotion);

    // Emotion-driven background tint
    void setEmotionSnapshotPath(const std::string& path);

    // Input hooks called from AppKit host
    void handleMouse(juce::MouseInputSource::MouseEventType type,
                     juce::Point<float> pos,
                     const juce::ModifierKeys& modifiers,
                     int clickCount);
    void handleScroll(juce::Point<float> pos, float deltaX, float deltaY, bool precise);
    void handleKeyDown(unsigned short keyCode, const juce::ModifierKeys& modifiers);
    void handleKeyUp(unsigned short keyCode, const juce::ModifierKeys& modifiers);

private:
    void timerCallback() override;
    void updatePlayhead();
    void updateEmotionTint();
    juce::Colour getEmotionTintColour(float valence) const;

    float playheadX_ = 0.0f;
    bool isDark_ = true;
    std::unique_ptr<juce::LookAndFeel> laf_;

    std::unique_ptr<MLVisualizationLayer> mlLayer_;
    bool mlVisEnabled_ {false};

    // Emotion tint state
    std::string emotionSnapshotPath_;
    float cachedValence_ {0.0f};
    juce::int64 lastSnapshotCheck_ {0};
    static constexpr juce::int64 snapshotCheckInterval_ = 500000; // microseconds (0.5s)
};
