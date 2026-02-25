#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_gui_extra/juce_gui_extra.h>
#include "plugin_processor.h"

namespace kelly {

//=============================================================================
// CUSTOM LOOK AND FEEL - Modern Glass Aesthetic
//=============================================================================

class KellyLookAndFeel : public juce::LookAndFeel_V4 {
public:
    KellyLookAndFeel();
    
    void drawRotarySlider(juce::Graphics& g, int x, int y, int width, int height,
                          float sliderPos, float rotaryStartAngle, float rotaryEndAngle,
                          juce::Slider& slider) override;
    
    void drawLinearSlider(juce::Graphics& g, int x, int y, int width, int height,
                          float sliderPos, float minSliderPos, float maxSliderPos,
                          juce::Slider::SliderStyle style, juce::Slider& slider) override;
    
    void drawButtonBackground(juce::Graphics& g, juce::Button& button,
                              const juce::Colour& backgroundColour,
                              bool shouldDrawButtonAsHighlighted,
                              bool shouldDrawButtonAsDown) override;
    
    void drawComboBox(juce::Graphics& g, int width, int height, bool isButtonDown,
                      int buttonX, int buttonY, int buttonW, int buttonH,
                      juce::ComboBox& box) override;
    
    void drawTextEditorOutline(juce::Graphics& g, int width, int height,
                               juce::TextEditor& textEditor) override;
    
    juce::Font getTextButtonFont(juce::TextButton&, int buttonHeight) override;
    juce::Font getComboBoxFont(juce::ComboBox&) override;
    juce::Font getLabelFont(juce::Label&) override;
    
private:
    juce::Colour accentColor{0xFF6366F1};      // Indigo
    juce::Colour accentColorAlt{0xFFF472B6};   // Pink
    juce::Colour surfaceColor{0xFF1E1E2E};     // Dark surface
    juce::Colour surfaceColorLight{0xFF2A2A40};
    juce::Colour textColor{0xFFE2E8F0};
    juce::Colour textColorMuted{0xFF94A3B8};
};

//=============================================================================
// GLASS PANEL COMPONENT
//=============================================================================

class GlassPanel : public juce::Component {
public:
    GlassPanel(const juce::String& title = "");
    void paint(juce::Graphics& g) override;
    void setTitle(const juce::String& newTitle) { title = newTitle; repaint(); }
    
private:
    juce::String title;
};

//=============================================================================
// EMOTION SIDE COMPONENT (Side A / Side B)
//=============================================================================

class EmotionSide : public juce::Component {
public:
    EmotionSide(const juce::String& sideName, bool isPrimary);
    void resized() override;
    void paint(juce::Graphics& g) override;
    
    juce::String getEmotionText() const { return emotionInput.getText(); }
    float getIntensity() const { return intensitySlider.getValue(); }
    void setEmotionText(const juce::String& text) { emotionInput.setText(text, false); }
    void setIntensity(float value) { intensitySlider.setValue(value, juce::sendNotification); }
    
    std::function<void()> onEmotionChanged;
    
private:
    juce::String sideName;
    bool isPrimary;
    
    juce::Label sideLabel;
    juce::Label emotionLabel;
    juce::TextEditor emotionInput;
    juce::Label intensityLabel;
    juce::Slider intensitySlider;
    
    void setupComponents();
};

//=============================================================================
// TRANSPORT BAR COMPONENT
//=============================================================================

class TransportBar : public juce::Component, private juce::Timer {
public:
    TransportBar();
    ~TransportBar() override;
    
    void resized() override;
    void paint(juce::Graphics& g) override;
    
    std::function<void()> onGenerate;
    std::function<void()> onPlay;
    std::function<void()> onStop;
    std::function<void()> onExport;
    
    void setPlaying(bool playing);
    void setProgress(float progress);
    
private:
    void timerCallback() override;
    
    juce::TextButton generateButton{"Generate"};
    juce::TextButton playButton{"▶"};
    juce::TextButton stopButton{"■"};
    juce::TextButton exportButton{"Export"};
    
    float progress = 0.0f;
    bool isPlaying = false;
    float pulsePhase = 0.0f;
};

//=============================================================================
// PARAMETER PANEL
//=============================================================================

class ParameterPanel : public juce::Component {
public:
    ParameterPanel();
    void resized() override;
    void paint(juce::Graphics& g) override;
    
    int getTempo() const { return static_cast<int>(tempoSlider.getValue()); }
    int getBars() const { return static_cast<int>(barsSlider.getValue()); }
    juce::String getKey() const { return keyCombo.getText(); }
    juce::String getMode() const { return modeCombo.getText(); }
    juce::String getGenre() const { return genreCombo.getText(); }
    
    void setTempo(int value) { tempoSlider.setValue(value, juce::dontSendNotification); }
    void setBars(int value) { barsSlider.setValue(value, juce::dontSendNotification); }
    
private:
    juce::Label tempoLabel{"", "BPM"};
    juce::Slider tempoSlider;
    
    juce::Label barsLabel{"", "BARS"};
    juce::Slider barsSlider;
    
    juce::Label keyLabel{"", "KEY"};
    juce::ComboBox keyCombo;
    
    juce::Label modeLabel{"", "MODE"};
    juce::ComboBox modeCombo;
    
    juce::Label genreLabel{"", "GENRE"};
    juce::ComboBox genreCombo;
    
    void setupComponents();
};

//=============================================================================
// WAVEFORM DISPLAY
//=============================================================================

class WaveformDisplay : public juce::Component, private juce::Timer {
public:
    WaveformDisplay();
    ~WaveformDisplay() override;
    
    void paint(juce::Graphics& g) override;
    void setPlaying(bool playing);
    void setProgress(float prog);
    
private:
    void timerCallback() override;
    
    std::vector<float> waveformData;
    float phase = 0.0f;
    float progress = 0.0f;
    bool isPlaying = false;
    
    void generateWaveform();
};

//=============================================================================
// MIDI DRAG COMPONENT
//=============================================================================

class MidiDragComponent : public juce::Component, public juce::DragAndDropContainer {
public:
    MidiDragComponent();
    void paint(juce::Graphics& g) override;
    void mouseDown(const juce::MouseEvent& e) override;
    void mouseDrag(const juce::MouseEvent& e) override;
    
    void setMidiFile(const juce::File& file) { midiFile = file; hasContent = file.existsAsFile(); repaint(); }
    bool hasMidiContent() const { return hasContent; }
    
private:
    juce::File midiFile;
    bool hasContent = false;
    bool isDragging = false;
};

//=============================================================================
// MAIN PLUGIN EDITOR
//=============================================================================

class PluginEditor : public juce::AudioProcessorEditor, private juce::Timer {
public:
    explicit PluginEditor(PluginProcessor&);
    ~PluginEditor() override;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    void timerCallback() override;
    void handleGenerate();
    void handlePlay();
    void handleStop();
    void handleExport();
    void handleDragToDAW();
    void updateFromProcessor();
    void createTempMidiFile();
    
    PluginProcessor& processor;
    KellyLookAndFeel lookAndFeel;
    
    // Header
    juce::Label titleLabel;
    juce::Label versionLabel;
    
    // Main panels
    GlassPanel emotionPanel{"EMOTIONAL INTENT"};
    EmotionSide sideA{"SIDE A — Where You Are", true};
    EmotionSide sideB{"SIDE B — Where You Want To Go", false};
    
    GlassPanel controlPanel{"GENERATION"};
    ParameterPanel parameterPanel;
    TransportBar transportBar;
    
    GlassPanel previewPanel{"PREVIEW"};
    WaveformDisplay waveformDisplay;
    MidiDragComponent midiDragComponent;
    
    // Temp file for drag
    juce::File tempMidiFile;
    
    // Background animation
    float bgPhase = 0.0f;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PluginEditor)
};

} // namespace kelly
