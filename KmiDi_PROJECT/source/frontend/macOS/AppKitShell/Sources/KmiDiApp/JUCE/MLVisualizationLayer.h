#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <array>

// Non-intrusive visualization layer for Kelly ML outputs.
// Renders soft hints on top of the timeline without committing edits.
class MLVisualizationLayer : public juce::Component
{
public:
    struct Data
    {
        std::array<float, 128> noteProbabilities {};
        std::array<float, 64>  chordProbabilities {};
        std::array<float, 32>  groove {};
        std::array<float, 16>  dynamics {};
        float valence {0.0f};
        float arousal {0.0f};
        float dominance {0.0f};
        float complexity {0.0f};
    };

    MLVisualizationLayer();
    ~MLVisualizationLayer() override = default;

    void paint (juce::Graphics& g) override;
    void resized() override;

    void setData (const Data& d);
    void clearData();

    void setZoom (float zoom);

    void setShowMelody (bool enable);
    void setShowHarmony (bool enable);
    void setShowGroove (bool enable);
    void setShowDynamics (bool enable);
    void setShowEmotion (bool enable);

    bool isShowingMelody() const noexcept   { return showMelody; }
    bool isShowingHarmony() const noexcept  { return showHarmony; }
    bool isShowingGroove() const noexcept   { return showGroove; }
    bool isShowingDynamics() const noexcept { return showDynamics; }
    bool isShowingEmotion() const noexcept  { return showEmotion; }

    void setDarkMode (bool isDark);

private:
    void rebuildPaths();
    void drawEmotionTint (juce::Graphics& g);
    void drawMelody (juce::Graphics& g);
    void drawHarmony (juce::Graphics& g);
    void drawGroove (juce::Graphics& g);
    void drawDynamics (juce::Graphics& g);

    Data data{};
    bool hasData {false};
    bool needsRebuild {true};
    bool isDark {true};

    bool showMelody {true};
    bool showHarmony {true};
    bool showGroove {true};
    bool showDynamics {true};
    bool showEmotion {true};

    float zoomLevel {1.0f}; // >1 = zoomed in; <1 = zoomed out

    juce::Path melodyPath;
    juce::Path harmonyPath;
    juce::Path dynamicsPath;
    juce::Array<juce::Line<float>> grooveLines;
};
