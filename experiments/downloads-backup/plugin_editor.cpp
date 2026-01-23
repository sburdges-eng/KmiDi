#include "plugin_editor.h"
#include <cmath>

namespace kelly {

//=============================================================================
// COLOR PALETTE
//=============================================================================

namespace Colors {
    const juce::Colour background{0xFF0F0F1A};
    const juce::Colour backgroundAlt{0xFF161625};
    const juce::Colour surface{0xFF1E1E2E};
    const juce::Colour surfaceLight{0xFF2A2A40};
    const juce::Colour surfaceLighter{0xFF3A3A55};
    
    const juce::Colour accent{0xFF6366F1};        // Indigo
    const juce::Colour accentAlt{0xFFF472B6};     // Pink
    const juce::Colour accentTertiary{0xFF22D3EE}; // Cyan
    
    const juce::Colour text{0xFFE2E8F0};
    const juce::Colour textMuted{0xFF94A3B8};
    const juce::Colour textDim{0xFF64748B};
    
    const juce::Colour success{0xFF10B981};
    const juce::Colour warning{0xFFFBBF24};
    const juce::Colour error{0xFFEF4444};
    
    const juce::Colour glassBorder{0x20FFFFFF};
    const juce::Colour glassHighlight{0x08FFFFFF};
}

//=============================================================================
// KELLY LOOK AND FEEL
//=============================================================================

KellyLookAndFeel::KellyLookAndFeel() {
    setColour(juce::TextEditor::backgroundColourId, Colors::surface);
    setColour(juce::TextEditor::textColourId, Colors::text);
    setColour(juce::TextEditor::outlineColourId, Colors::surfaceLighter);
    setColour(juce::TextEditor::focusedOutlineColourId, Colors::accent);
    setColour(juce::TextEditor::highlightColourId, Colors::accent.withAlpha(0.3f));
    
    setColour(juce::ComboBox::backgroundColourId, Colors::surface);
    setColour(juce::ComboBox::textColourId, Colors::text);
    setColour(juce::ComboBox::outlineColourId, Colors::surfaceLighter);
    setColour(juce::ComboBox::arrowColourId, Colors::textMuted);
    
    setColour(juce::PopupMenu::backgroundColourId, Colors::surface);
    setColour(juce::PopupMenu::textColourId, Colors::text);
    setColour(juce::PopupMenu::highlightedBackgroundColourId, Colors::accent.withAlpha(0.2f));
    setColour(juce::PopupMenu::highlightedTextColourId, Colors::text);
    
    setColour(juce::Label::textColourId, Colors::text);
    
    setColour(juce::Slider::backgroundColourId, Colors::surfaceLight);
    setColour(juce::Slider::trackColourId, Colors::accent);
    setColour(juce::Slider::thumbColourId, Colors::text);
}

void KellyLookAndFeel::drawRotarySlider(juce::Graphics& g, int x, int y, int width, int height,
                                         float sliderPos, float rotaryStartAngle, float rotaryEndAngle,
                                         juce::Slider& slider) {
    auto radius = (float)juce::jmin(width / 2, height / 2) - 8.0f;
    auto centreX = (float)x + (float)width * 0.5f;
    auto centreY = (float)y + (float)height * 0.5f;
    auto rx = centreX - radius;
    auto ry = centreY - radius;
    auto rw = radius * 2.0f;
    auto angle = rotaryStartAngle + sliderPos * (rotaryEndAngle - rotaryStartAngle);

    // Background track
    g.setColour(Colors::surfaceLight);
    g.fillEllipse(rx, ry, rw, rw);
    
    // Outer glow when focused
    if (slider.hasKeyboardFocus(false)) {
        g.setColour(Colors::accent.withAlpha(0.3f));
        g.drawEllipse(rx - 2, ry - 2, rw + 4, rw + 4, 2.0f);
    }

    // Value arc
    juce::Path arcPath;
    arcPath.addCentredArc(centreX, centreY, radius - 4, radius - 4,
                          0.0f, rotaryStartAngle, angle, true);
    
    juce::ColourGradient gradient(Colors::accent, centreX, centreY - radius,
                                   Colors::accentAlt, centreX, centreY + radius, false);
    g.setGradientFill(gradient);
    g.strokePath(arcPath, juce::PathStrokeType(4.0f, juce::PathStrokeType::curved,
                                                juce::PathStrokeType::rounded));

    // Thumb dot
    juce::Point<float> thumbPoint(centreX + (radius - 12) * std::cos(angle - juce::MathConstants<float>::halfPi),
                                   centreY + (radius - 12) * std::sin(angle - juce::MathConstants<float>::halfPi));
    
    g.setColour(Colors::text);
    g.fillEllipse(thumbPoint.x - 4, thumbPoint.y - 4, 8, 8);
}

void KellyLookAndFeel::drawLinearSlider(juce::Graphics& g, int x, int y, int width, int height,
                                         float sliderPos, float minSliderPos, float maxSliderPos,
                                         juce::Slider::SliderStyle style, juce::Slider& slider) {
    auto trackWidth = 6.0f;
    auto isHorizontal = style == juce::Slider::LinearHorizontal || style == juce::Slider::LinearBar;
    
    if (isHorizontal) {
        auto trackY = (float)y + (float)height * 0.5f - trackWidth * 0.5f;
        
        // Background track
        g.setColour(Colors::surfaceLight);
        g.fillRoundedRectangle((float)x, trackY, (float)width, trackWidth, trackWidth * 0.5f);
        
        // Value track with gradient
        auto fillWidth = sliderPos - (float)x;
        if (fillWidth > 0) {
            juce::ColourGradient gradient(Colors::accent, (float)x, trackY,
                                           Colors::accentAlt, sliderPos, trackY, false);
            g.setGradientFill(gradient);
            g.fillRoundedRectangle((float)x, trackY, fillWidth, trackWidth, trackWidth * 0.5f);
        }
        
        // Thumb
        auto thumbSize = 16.0f;
        auto thumbY = (float)y + (float)height * 0.5f - thumbSize * 0.5f;
        
        g.setColour(Colors::text);
        g.fillEllipse(sliderPos - thumbSize * 0.5f, thumbY, thumbSize, thumbSize);
        
        // Inner dot
        g.setColour(Colors::accent);
        g.fillEllipse(sliderPos - 3, thumbY + thumbSize * 0.5f - 3, 6, 6);
    }
    else {
        auto trackX = (float)x + (float)width * 0.5f - trackWidth * 0.5f;
        
        // Background track
        g.setColour(Colors::surfaceLight);
        g.fillRoundedRectangle(trackX, (float)y, trackWidth, (float)height, trackWidth * 0.5f);
        
        // Value track
        auto fillHeight = (float)y + (float)height - sliderPos;
        if (fillHeight > 0) {
            juce::ColourGradient gradient(Colors::accentAlt, trackX, (float)y + (float)height,
                                           Colors::accent, trackX, sliderPos, false);
            g.setGradientFill(gradient);
            g.fillRoundedRectangle(trackX, sliderPos, trackWidth, fillHeight, trackWidth * 0.5f);
        }
        
        // Thumb
        auto thumbSize = 16.0f;
        auto thumbX = (float)x + (float)width * 0.5f - thumbSize * 0.5f;
        
        g.setColour(Colors::text);
        g.fillEllipse(thumbX, sliderPos - thumbSize * 0.5f, thumbSize, thumbSize);
    }
}

void KellyLookAndFeel::drawButtonBackground(juce::Graphics& g, juce::Button& button,
                                             const juce::Colour& backgroundColour,
                                             bool shouldDrawButtonAsHighlighted,
                                             bool shouldDrawButtonAsDown) {
    auto bounds = button.getLocalBounds().toFloat().reduced(1.0f);
    auto cornerSize = 8.0f;
    
    juce::Colour bgColor = backgroundColour;
    
    if (shouldDrawButtonAsDown) {
        bgColor = Colors::accent.darker(0.2f);
    }
    else if (shouldDrawButtonAsHighlighted) {
        bgColor = Colors::accent.brighter(0.1f);
    }
    
    // Button background with gradient
    if (button.getToggleState() || shouldDrawButtonAsDown) {
        juce::ColourGradient gradient(Colors::accent, bounds.getX(), bounds.getY(),
                                       Colors::accentAlt, bounds.getRight(), bounds.getBottom(), true);
        g.setGradientFill(gradient);
    }
    else {
        g.setColour(Colors::surfaceLight);
    }
    
    g.fillRoundedRectangle(bounds, cornerSize);
    
    // Border
    if (shouldDrawButtonAsHighlighted || button.hasKeyboardFocus(false)) {
        g.setColour(Colors::accent.withAlpha(0.5f));
        g.drawRoundedRectangle(bounds, cornerSize, 1.5f);
    }
    else {
        g.setColour(Colors::surfaceLighter);
        g.drawRoundedRectangle(bounds, cornerSize, 1.0f);
    }
}

void KellyLookAndFeel::drawComboBox(juce::Graphics& g, int width, int height, bool isButtonDown,
                                     int buttonX, int buttonY, int buttonW, int buttonH,
                                     juce::ComboBox& box) {
    auto bounds = juce::Rectangle<float>(0, 0, (float)width, (float)height).reduced(1.0f);
    auto cornerSize = 6.0f;
    
    g.setColour(Colors::surface);
    g.fillRoundedRectangle(bounds, cornerSize);
    
    g.setColour(box.hasKeyboardFocus(false) ? Colors::accent : Colors::surfaceLighter);
    g.drawRoundedRectangle(bounds, cornerSize, 1.0f);
    
    // Arrow
    auto arrowZone = juce::Rectangle<float>((float)buttonX, (float)buttonY, (float)buttonW, (float)buttonH);
    juce::Path arrow;
    arrow.addTriangle(arrowZone.getCentreX() - 4, arrowZone.getCentreY() - 2,
                      arrowZone.getCentreX() + 4, arrowZone.getCentreY() - 2,
                      arrowZone.getCentreX(), arrowZone.getCentreY() + 3);
    g.setColour(Colors::textMuted);
    g.fillPath(arrow);
}

void KellyLookAndFeel::drawTextEditorOutline(juce::Graphics& g, int width, int height,
                                              juce::TextEditor& textEditor) {
    auto bounds = juce::Rectangle<float>(0, 0, (float)width, (float)height).reduced(0.5f);
    auto cornerSize = 6.0f;
    
    g.setColour(textEditor.hasKeyboardFocus(true) ? Colors::accent : Colors::surfaceLighter);
    g.drawRoundedRectangle(bounds, cornerSize, 1.0f);
}

juce::Font KellyLookAndFeel::getTextButtonFont(juce::TextButton&, int buttonHeight) {
    return juce::Font(juce::FontOptions().withHeight((float)buttonHeight * 0.45f).withStyle("Bold"));
}

juce::Font KellyLookAndFeel::getComboBoxFont(juce::ComboBox&) {
    return juce::Font(juce::FontOptions().withHeight(14.0f));
}

juce::Font KellyLookAndFeel::getLabelFont(juce::Label&) {
    return juce::Font(juce::FontOptions().withHeight(13.0f));
}

//=============================================================================
// GLASS PANEL
//=============================================================================

GlassPanel::GlassPanel(const juce::String& t) : title(t) {
    setOpaque(false);
}

void GlassPanel::paint(juce::Graphics& g) {
    auto bounds = getLocalBounds().toFloat();
    auto cornerSize = 12.0f;
    
    // Glass background
    g.setColour(Colors::surface.withAlpha(0.7f));
    g.fillRoundedRectangle(bounds, cornerSize);
    
    // Top highlight
    juce::ColourGradient highlightGrad(Colors::glassHighlight, bounds.getX(), bounds.getY(),
                                        juce::Colours::transparentBlack, bounds.getX(), bounds.getY() + 60,
                                        false);
    g.setGradientFill(highlightGrad);
    g.fillRoundedRectangle(bounds.removeFromTop(60), cornerSize);
    
    // Border
    g.setColour(Colors::glassBorder);
    g.drawRoundedRectangle(getLocalBounds().toFloat().reduced(0.5f), cornerSize, 1.0f);
    
    // Title
    if (title.isNotEmpty()) {
        g.setColour(Colors::textMuted);
        g.setFont(juce::Font(juce::FontOptions().withHeight(11.0f).withStyle("Bold")));
        g.drawText(title, 16, 12, getWidth() - 32, 16, juce::Justification::centredLeft);
    }
}

//=============================================================================
// EMOTION SIDE
//=============================================================================

EmotionSide::EmotionSide(const juce::String& name, bool primary)
    : sideName(name), isPrimary(primary) {
    setupComponents();
}

void EmotionSide::setupComponents() {
    // Side label
    sideLabel.setText(sideName, juce::dontSendNotification);
    sideLabel.setFont(juce::Font(juce::FontOptions().withHeight(12.0f).withStyle("Bold")));
    sideLabel.setColour(juce::Label::textColourId, isPrimary ? Colors::accent : Colors::accentAlt);
    addAndMakeVisible(sideLabel);
    
    // Emotion input label
    emotionLabel.setText("Describe your emotion:", juce::dontSendNotification);
    emotionLabel.setFont(juce::Font(juce::FontOptions().withHeight(11.0f)));
    emotionLabel.setColour(juce::Label::textColourId, Colors::textMuted);
    addAndMakeVisible(emotionLabel);
    
    // Emotion text input
    emotionInput.setMultiLine(true);
    emotionInput.setReturnKeyStartsNewLine(false);
    emotionInput.setTextToShowWhenEmpty(isPrimary 
        ? "e.g., Anxious, like waiting for bad news..."
        : "e.g., Hopeful, like finally seeing light...", 
        Colors::textDim);
    emotionInput.onTextChange = [this]() {
        if (onEmotionChanged) onEmotionChanged();
    };
    addAndMakeVisible(emotionInput);
    
    // Intensity label
    intensityLabel.setText("INTENSITY", juce::dontSendNotification);
    intensityLabel.setFont(juce::Font(juce::FontOptions().withHeight(10.0f).withStyle("Bold")));
    intensityLabel.setColour(juce::Label::textColourId, Colors::textMuted);
    addAndMakeVisible(intensityLabel);
    
    // Intensity slider
    intensitySlider.setSliderStyle(juce::Slider::LinearHorizontal);
    intensitySlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 40, 20);
    intensitySlider.setRange(0.0, 1.0, 0.01);
    intensitySlider.setValue(isPrimary ? 0.7 : 0.5);
    intensitySlider.setColour(juce::Slider::textBoxTextColourId, Colors::text);
    intensitySlider.setColour(juce::Slider::textBoxBackgroundColourId, Colors::surface);
    intensitySlider.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    intensitySlider.onValueChange = [this]() {
        if (onEmotionChanged) onEmotionChanged();
    };
    addAndMakeVisible(intensitySlider);
}

void EmotionSide::resized() {
    auto bounds = getLocalBounds().reduced(12, 8);
    
    sideLabel.setBounds(bounds.removeFromTop(18));
    bounds.removeFromTop(8);
    
    emotionLabel.setBounds(bounds.removeFromTop(16));
    bounds.removeFromTop(4);
    
    emotionInput.setBounds(bounds.removeFromTop(60));
    bounds.removeFromTop(12);
    
    auto sliderRow = bounds.removeFromTop(24);
    intensityLabel.setBounds(sliderRow.removeFromLeft(70));
    intensitySlider.setBounds(sliderRow);
}

void EmotionSide::paint(juce::Graphics& g) {
    auto bounds = getLocalBounds().toFloat();
    
    // Subtle gradient background based on side
    juce::ColourGradient grad(
        (isPrimary ? Colors::accent : Colors::accentAlt).withAlpha(0.05f),
        isPrimary ? bounds.getX() : bounds.getRight(),
        bounds.getY(),
        juce::Colours::transparentBlack,
        isPrimary ? bounds.getRight() : bounds.getX(),
        bounds.getBottom(),
        false
    );
    g.setGradientFill(grad);
    g.fillRoundedRectangle(bounds, 8.0f);
    
    // Side indicator line
    g.setColour(isPrimary ? Colors::accent : Colors::accentAlt);
    if (isPrimary) {
        g.fillRoundedRectangle(0, 0, 3, (float)getHeight(), 1.5f);
    } else {
        g.fillRoundedRectangle((float)getWidth() - 3, 0, 3, (float)getHeight(), 1.5f);
    }
}

//=============================================================================
// TRANSPORT BAR
//=============================================================================

TransportBar::TransportBar() {
    generateButton.onClick = [this]() { if (onGenerate) onGenerate(); };
    playButton.onClick = [this]() { if (onPlay) onPlay(); };
    stopButton.onClick = [this]() { if (onStop) onStop(); };
    exportButton.onClick = [this]() { if (onExport) onExport(); };
    
    addAndMakeVisible(generateButton);
    addAndMakeVisible(playButton);
    addAndMakeVisible(stopButton);
    addAndMakeVisible(exportButton);
    
    startTimerHz(30);
}

TransportBar::~TransportBar() {
    stopTimer();
}

void TransportBar::timerCallback() {
    if (isPlaying) {
        pulsePhase += 0.1f;
        repaint();
    }
}

void TransportBar::resized() {
    auto bounds = getLocalBounds().reduced(8, 4);
    
    auto buttonHeight = bounds.getHeight();
    auto buttonWidth = 90;
    auto smallButtonWidth = 40;
    auto spacing = 8;
    
    generateButton.setBounds(bounds.removeFromLeft(buttonWidth));
    bounds.removeFromLeft(spacing);
    
    playButton.setBounds(bounds.removeFromLeft(smallButtonWidth));
    bounds.removeFromLeft(4);
    stopButton.setBounds(bounds.removeFromLeft(smallButtonWidth));
    
    exportButton.setBounds(bounds.removeFromRight(buttonWidth));
}

void TransportBar::paint(juce::Graphics& g) {
    auto bounds = getLocalBounds().toFloat();
    
    // Progress bar background (between buttons)
    auto progressBounds = bounds.reduced(110, 8);
    progressBounds.removeFromLeft(100);
    progressBounds.removeFromRight(100);
    
    if (progressBounds.getWidth() > 20) {
        g.setColour(Colors::surfaceLight);
        g.fillRoundedRectangle(progressBounds, 4.0f);
        
        if (progress > 0 || isPlaying) {
            auto fillWidth = progressBounds.getWidth() * progress;
            
            juce::ColourGradient grad(Colors::accent, progressBounds.getX(), progressBounds.getY(),
                                       Colors::accentAlt, progressBounds.getRight(), progressBounds.getY(),
                                       false);
            g.setGradientFill(grad);
            g.fillRoundedRectangle(progressBounds.getX(), progressBounds.getY(),
                                   fillWidth, progressBounds.getHeight(), 4.0f);
            
            // Animated pulse when playing
            if (isPlaying) {
                float pulse = 0.5f + 0.5f * std::sin(pulsePhase);
                g.setColour(Colors::accent.withAlpha(0.3f * pulse));
                g.fillRoundedRectangle(progressBounds.getX() + fillWidth - 4,
                                       progressBounds.getY() - 2,
                                       8, progressBounds.getHeight() + 4, 4.0f);
            }
        }
    }
}

void TransportBar::setPlaying(bool playing) {
    isPlaying = playing;
    playButton.setButtonText(playing ? "⏸" : "▶");
    repaint();
}

void TransportBar::setProgress(float prog) {
    progress = juce::jlimit(0.0f, 1.0f, prog);
    repaint();
}

//=============================================================================
// PARAMETER PANEL
//=============================================================================

ParameterPanel::ParameterPanel() {
    setupComponents();
}

void ParameterPanel::setupComponents() {
    // Tempo
    tempoLabel.setFont(juce::Font(juce::FontOptions().withHeight(10.0f).withStyle("Bold")));
    tempoLabel.setColour(juce::Label::textColourId, Colors::textMuted);
    tempoLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(tempoLabel);
    
    tempoSlider.setSliderStyle(juce::Slider::RotaryVerticalDrag);
    tempoSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 50, 18);
    tempoSlider.setRange(40, 200, 1);
    tempoSlider.setValue(120);
    tempoSlider.setColour(juce::Slider::textBoxTextColourId, Colors::text);
    tempoSlider.setColour(juce::Slider::textBoxBackgroundColourId, juce::Colours::transparentBlack);
    tempoSlider.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    addAndMakeVisible(tempoSlider);
    
    // Bars
    barsLabel.setFont(juce::Font(juce::FontOptions().withHeight(10.0f).withStyle("Bold")));
    barsLabel.setColour(juce::Label::textColourId, Colors::textMuted);
    barsLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(barsLabel);
    
    barsSlider.setSliderStyle(juce::Slider::RotaryVerticalDrag);
    barsSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 50, 18);
    barsSlider.setRange(1, 32, 1);
    barsSlider.setValue(8);
    barsSlider.setColour(juce::Slider::textBoxTextColourId, Colors::text);
    barsSlider.setColour(juce::Slider::textBoxBackgroundColourId, juce::Colours::transparentBlack);
    barsSlider.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    addAndMakeVisible(barsSlider);
    
    // Key
    keyLabel.setFont(juce::Font(juce::FontOptions().withHeight(10.0f).withStyle("Bold")));
    keyLabel.setColour(juce::Label::textColourId, Colors::textMuted);
    addAndMakeVisible(keyLabel);
    
    keyCombo.addItemList({"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"}, 1);
    keyCombo.setSelectedId(1);
    addAndMakeVisible(keyCombo);
    
    // Mode
    modeLabel.setFont(juce::Font(juce::FontOptions().withHeight(10.0f).withStyle("Bold")));
    modeLabel.setColour(juce::Label::textColourId, Colors::textMuted);
    addAndMakeVisible(modeLabel);
    
    modeCombo.addItemList({"Major", "Minor", "Dorian", "Phrygian", "Lydian", "Mixolydian", "Aeolian", "Locrian"}, 1);
    modeCombo.setSelectedId(2);
    addAndMakeVisible(modeCombo);
    
    // Genre
    genreLabel.setFont(juce::Font(juce::FontOptions().withHeight(10.0f).withStyle("Bold")));
    genreLabel.setColour(juce::Label::textColourId, Colors::textMuted);
    addAndMakeVisible(genreLabel);
    
    genreCombo.addItemList({"Ambient", "Cinematic", "Electronic", "Folk", "Jazz", "Lo-Fi", "Neo-Soul", "Pop", "R&B", "Rock"}, 1);
    genreCombo.setSelectedId(1);
    addAndMakeVisible(genreCombo);
}

void ParameterPanel::resized() {
    auto bounds = getLocalBounds().reduced(8);
    
    // Knobs on top row
    auto knobRow = bounds.removeFromTop(90);
    auto knobWidth = knobRow.getWidth() / 2;
    
    auto tempoBounds = knobRow.removeFromLeft(knobWidth);
    tempoLabel.setBounds(tempoBounds.removeFromTop(16));
    tempoSlider.setBounds(tempoBounds);
    
    auto barsBounds = knobRow;
    barsLabel.setBounds(barsBounds.removeFromTop(16));
    barsSlider.setBounds(barsBounds);
    
    bounds.removeFromTop(8);
    
    // Combos below
    auto comboHeight = 28;
    auto labelHeight = 14;
    
    keyLabel.setBounds(bounds.removeFromTop(labelHeight));
    keyCombo.setBounds(bounds.removeFromTop(comboHeight));
    bounds.removeFromTop(8);
    
    modeLabel.setBounds(bounds.removeFromTop(labelHeight));
    modeCombo.setBounds(bounds.removeFromTop(comboHeight));
    bounds.removeFromTop(8);
    
    genreLabel.setBounds(bounds.removeFromTop(labelHeight));
    genreCombo.setBounds(bounds.removeFromTop(comboHeight));
}

void ParameterPanel::paint(juce::Graphics& g) {
    // Background handled by parent
}

//=============================================================================
// WAVEFORM DISPLAY
//=============================================================================

WaveformDisplay::WaveformDisplay() {
    generateWaveform();
    startTimerHz(30);
}

WaveformDisplay::~WaveformDisplay() {
    stopTimer();
}

void WaveformDisplay::generateWaveform() {
    waveformData.resize(256);
    for (size_t i = 0; i < waveformData.size(); ++i) {
        float x = (float)i / waveformData.size();
        waveformData[i] = 0.3f * std::sin(x * 8 * juce::MathConstants<float>::pi)
                        + 0.2f * std::sin(x * 16 * juce::MathConstants<float>::pi)
                        + 0.1f * std::sin(x * 32 * juce::MathConstants<float>::pi);
    }
}

void WaveformDisplay::timerCallback() {
    if (isPlaying) {
        phase += 0.02f;
        repaint();
    }
}

void WaveformDisplay::paint(juce::Graphics& g) {
    auto bounds = getLocalBounds().toFloat().reduced(4);
    
    // Background
    g.setColour(Colors::background);
    g.fillRoundedRectangle(bounds, 6.0f);
    
    // Grid lines
    g.setColour(Colors::surfaceLight.withAlpha(0.3f));
    for (int i = 1; i < 4; ++i) {
        float y = bounds.getY() + bounds.getHeight() * (float)i / 4.0f;
        g.drawHorizontalLine((int)y, bounds.getX(), bounds.getRight());
    }
    for (int i = 1; i < 8; ++i) {
        float x = bounds.getX() + bounds.getWidth() * (float)i / 8.0f;
        g.drawVerticalLine((int)x, bounds.getY(), bounds.getBottom());
    }
    
    // Waveform
    if (!waveformData.empty()) {
        juce::Path waveformPath;
        auto midY = bounds.getCentreY();
        auto height = bounds.getHeight() * 0.35f;
        
        waveformPath.startNewSubPath(bounds.getX(), midY);
        
        for (size_t i = 0; i < waveformData.size(); ++i) {
            float x = bounds.getX() + (float)i / waveformData.size() * bounds.getWidth();
            float animatedPhase = phase + (float)i * 0.02f;
            float amplitude = isPlaying ? (0.8f + 0.2f * std::sin(animatedPhase * 3)) : 0.6f;
            float y = midY + waveformData[i] * height * amplitude;
            waveformPath.lineTo(x, y);
        }
        
        juce::ColourGradient waveGrad(Colors::accent, bounds.getX(), midY,
                                       Colors::accentAlt, bounds.getRight(), midY, false);
        g.setGradientFill(waveGrad);
        g.strokePath(waveformPath, juce::PathStrokeType(2.0f));
        
        // Glow effect
        g.setColour(Colors::accent.withAlpha(0.2f));
        g.strokePath(waveformPath, juce::PathStrokeType(6.0f));
    }
    
    // Playhead
    if (isPlaying || progress > 0) {
        float playheadX = bounds.getX() + bounds.getWidth() * progress;
        g.setColour(Colors::text);
        g.drawVerticalLine((int)playheadX, bounds.getY(), bounds.getBottom());
        g.fillEllipse(playheadX - 4, bounds.getY() - 4, 8, 8);
        g.fillEllipse(playheadX - 4, bounds.getBottom() - 4, 8, 8);
    }
    
    // Border
    g.setColour(Colors::surfaceLighter);
    g.drawRoundedRectangle(bounds, 6.0f, 1.0f);
}

void WaveformDisplay::setPlaying(bool playing) {
    isPlaying = playing;
}

void WaveformDisplay::setProgress(float prog) {
    progress = prog;
}

//=============================================================================
// MIDI DRAG COMPONENT
//=============================================================================

MidiDragComponent::MidiDragComponent() {
    setMouseCursor(juce::MouseCursor::DraggingHandCursor);
}

void MidiDragComponent::paint(juce::Graphics& g) {
    auto bounds = getLocalBounds().toFloat();
    auto cornerSize = 8.0f;
    
    // Background
    if (hasContent) {
        juce::ColourGradient grad(Colors::accent.withAlpha(0.2f), bounds.getX(), bounds.getY(),
                                   Colors::accentAlt.withAlpha(0.2f), bounds.getRight(), bounds.getBottom(), true);
        g.setGradientFill(grad);
    } else {
        g.setColour(Colors::surfaceLight.withAlpha(0.5f));
    }
    g.fillRoundedRectangle(bounds, cornerSize);
    
    // Border
    g.setColour(hasContent ? Colors::accent : Colors::surfaceLighter);
    g.drawRoundedRectangle(bounds.reduced(0.5f), cornerSize, 1.0f);
    
    // Icon and text
    auto centerY = bounds.getCentreY();
    
    if (hasContent) {
        // MIDI icon
        g.setColour(Colors::accent);
        g.fillRoundedRectangle(bounds.getCentreX() - 20, centerY - 12, 16, 24, 3);
        g.setColour(Colors::surface);
        g.fillEllipse(bounds.getCentreX() - 18, centerY - 6, 5, 5);
        g.fillEllipse(bounds.getCentreX() - 18, centerY + 2, 5, 5);
        g.fillEllipse(bounds.getCentreX() - 11, centerY - 2, 5, 5);
        
        // Text
        g.setColour(Colors::text);
        g.setFont(juce::Font(juce::FontOptions().withHeight(11.0f).withStyle("Bold")));
        g.drawText("DRAG TO TRACK", bounds.getCentreX() + 4, centerY - 8, 80, 16, juce::Justification::centredLeft);
        
        g.setColour(Colors::textMuted);
        g.setFont(juce::Font(juce::FontOptions().withHeight(9.0f)));
        g.drawText("Drop on Logic timeline", bounds.getCentreX() + 4, centerY + 4, 100, 12, juce::Justification::centredLeft);
    } else {
        g.setColour(Colors::textDim);
        g.setFont(juce::Font(juce::FontOptions().withHeight(11.0f)));
        g.drawText("Generate MIDI first", bounds, juce::Justification::centred);
    }
}

void MidiDragComponent::mouseDown(const juce::MouseEvent& e) {
    if (hasContent && midiFile.existsAsFile()) {
        isDragging = true;
    }
}

void MidiDragComponent::mouseDrag(const juce::MouseEvent& e) {
    if (isDragging && hasContent && midiFile.existsAsFile()) {
        isDragging = false;
        
        juce::StringArray files;
        files.add(midiFile.getFullPathName());
        performExternalDragDropOfFiles(files, false, this, nullptr);
    }
}

//=============================================================================
// MAIN PLUGIN EDITOR
//=============================================================================

PluginEditor::PluginEditor(PluginProcessor& p)
    : AudioProcessorEditor(&p), processor(p) {
    
    setLookAndFeel(&lookAndFeel);
    
    // Title
    titleLabel.setText("KELLY", juce::dontSendNotification);
    titleLabel.setFont(juce::Font(juce::FontOptions().withHeight(28.0f).withStyle("Bold")));
    titleLabel.setColour(juce::Label::textColourId, Colors::text);
    addAndMakeVisible(titleLabel);
    
    // Version
    versionLabel.setText("Emotion Processor v1.0", juce::dontSendNotification);
    versionLabel.setFont(juce::Font(juce::FontOptions().withHeight(11.0f)));
    versionLabel.setColour(juce::Label::textColourId, Colors::textMuted);
    addAndMakeVisible(versionLabel);
    
    // Emotion panel with sides
    addAndMakeVisible(emotionPanel);
    addAndMakeVisible(sideA);
    addAndMakeVisible(sideB);
    
    // Control panel
    addAndMakeVisible(controlPanel);
    addAndMakeVisible(parameterPanel);
    
    // Transport
    transportBar.onGenerate = [this]() { handleGenerate(); };
    transportBar.onPlay = [this]() { handlePlay(); };
    transportBar.onStop = [this]() { handleStop(); };
    transportBar.onExport = [this]() { handleExport(); };
    addAndMakeVisible(transportBar);
    
    // Preview panel
    addAndMakeVisible(previewPanel);
    addAndMakeVisible(waveformDisplay);
    addAndMakeVisible(midiDragComponent);
    
    // Create temp directory for MIDI files
    tempMidiFile = juce::File::getSpecialLocation(juce::File::tempDirectory)
                       .getChildFile("kelly_generated.mid");
    
    setSize(800, 650);
    startTimerHz(30);
}

PluginEditor::~PluginEditor() {
    stopTimer();
    setLookAndFeel(nullptr);
}

void PluginEditor::timerCallback() {
    bgPhase += 0.01f;
    repaint();
    updateFromProcessor();
}

void PluginEditor::paint(juce::Graphics& g) {
    // Animated gradient background
    auto bounds = getLocalBounds().toFloat();
    
    // Base gradient
    juce::ColourGradient bgGrad(Colors::background, bounds.getX(), bounds.getY(),
                                 Colors::backgroundAlt, bounds.getRight(), bounds.getBottom(), true);
    g.setGradientFill(bgGrad);
    g.fillRect(bounds);
    
    // Subtle animated orbs
    float orbX1 = bounds.getCentreX() + std::cos(bgPhase) * 100;
    float orbY1 = bounds.getCentreY() + std::sin(bgPhase * 0.7f) * 80;
    juce::ColourGradient orb1(Colors::accent.withAlpha(0.08f), orbX1, orbY1,
                               juce::Colours::transparentBlack, orbX1, orbY1 + 200, true);
    g.setGradientFill(orb1);
    g.fillEllipse(orbX1 - 150, orbY1 - 150, 300, 300);
    
    float orbX2 = bounds.getCentreX() + std::cos(bgPhase * 1.3f + 2) * 120;
    float orbY2 = bounds.getCentreY() + std::sin(bgPhase * 0.5f + 1) * 100;
    juce::ColourGradient orb2(Colors::accentAlt.withAlpha(0.05f), orbX2, orbY2,
                               juce::Colours::transparentBlack, orbX2, orbY2 + 180, true);
    g.setGradientFill(orb2);
    g.fillEllipse(orbX2 - 120, orbY2 - 120, 240, 240);
    
    // Noise/grain overlay (subtle)
    juce::Random rng;
    g.setColour(juce::Colours::white.withAlpha(0.008f));
    for (int i = 0; i < 50; ++i) {
        float x = rng.nextFloat() * bounds.getWidth();
        float y = rng.nextFloat() * bounds.getHeight();
        g.fillRect(x, y, 1.0f, 1.0f);
    }
}

void PluginEditor::resized() {
    auto bounds = getLocalBounds().reduced(20);
    
    // Header
    auto headerBounds = bounds.removeFromTop(50);
    titleLabel.setBounds(headerBounds.removeFromLeft(100));
    versionLabel.setBounds(headerBounds.removeFromLeft(200).withTrimmedTop(12));
    
    bounds.removeFromTop(10);
    
    // Main content area - two columns
    auto leftColumn = bounds.removeFromLeft(bounds.getWidth() * 0.6f - 10);
    bounds.removeFromLeft(20);
    auto rightColumn = bounds;
    
    // Left column: Emotion panel
    emotionPanel.setBounds(leftColumn);
    
    auto emotionInner = leftColumn.reduced(12, 36);
    auto sideHeight = (emotionInner.getHeight() - 20) / 2;
    
    sideA.setBounds(emotionInner.removeFromTop(sideHeight));
    emotionInner.removeFromTop(20);
    sideB.setBounds(emotionInner);
    
    // Right column: Parameters and preview
    auto paramHeight = 240;
    controlPanel.setBounds(rightColumn.removeFromTop(paramHeight));
    parameterPanel.setBounds(controlPanel.getBounds().reduced(12, 36));
    
    rightColumn.removeFromTop(12);
    
    // Transport bar
    transportBar.setBounds(rightColumn.removeFromTop(44));
    
    rightColumn.removeFromTop(12);
    
    // Preview - now includes waveform and drag area
    previewPanel.setBounds(rightColumn);
    auto previewInner = previewPanel.getBounds().reduced(12, 36);
    waveformDisplay.setBounds(previewInner.removeFromTop(70));
    previewInner.removeFromTop(8);
    midiDragComponent.setBounds(previewInner);
}

void PluginEditor::handleGenerate() {
    // Get values from UI and update processor
    processor.sideA.description = sideA.getEmotionText();
    processor.sideA.intensity = sideA.getIntensity();
    processor.sideB.description = sideB.getEmotionText();
    processor.sideB.intensity = sideB.getIntensity();
    
    processor.tempo.store(parameterPanel.getTempo());
    processor.bars.store(parameterPanel.getBars());
    processor.keyRoot = parameterPanel.getKey();
    processor.mode = parameterPanel.getMode();
    processor.genre = parameterPanel.getGenre();
    
    // Generate
    processor.generateFromEmotion();
    
    // Create temp MIDI file for drag-to-DAW
    createTempMidiFile();
    
    transportBar.setProgress(0.0f);
    DBG("Generate clicked - " << processor.generatedNotes.size() << " notes created");
}

void PluginEditor::createTempMidiFile() {
    if (processor.exportMidi(tempMidiFile)) {
        midiDragComponent.setMidiFile(tempMidiFile);
        DBG("Created temp MIDI file: " << tempMidiFile.getFullPathName());
    }
}

void PluginEditor::handlePlay() {
    if (processor.generatedNotes.empty()) {
        DBG("No MIDI to play - generate first");
        return;
    }
    
    processor.startPlayback();
    transportBar.setPlaying(true);
    waveformDisplay.setPlaying(true);
    DBG("Play clicked");
}

void PluginEditor::handleStop() {
    processor.stopPlayback();
    transportBar.setPlaying(false);
    waveformDisplay.setPlaying(false);
    transportBar.setProgress(0.0f);
    waveformDisplay.setProgress(0.0f);
    DBG("Stop clicked");
}

void PluginEditor::handleDragToDAW() {
    // Handled by MidiDragComponent
}

void PluginEditor::handleExport() {
    if (processor.generatedNotes.empty()) {
        DBG("No MIDI to export - generate first");
        return;
    }
    
    auto chooser = std::make_unique<juce::FileChooser>(
        "Export MIDI File",
        juce::File::getSpecialLocation(juce::File::userDesktopDirectory).getChildFile("kelly_export.mid"),
        "*.mid"
    );
    
    auto flags = juce::FileBrowserComponent::saveMode | juce::FileBrowserComponent::canSelectFiles;
    
    chooser->launchAsync(flags, [this](const juce::FileChooser& fc) {
        auto file = fc.getResult();
        if (file != juce::File{}) {
            if (processor.exportMidi(file)) {
                DBG("Exported MIDI to: " << file.getFullPathName());
            }
        }
    });
    
    // prevent chooser from being deleted until dialog closes
    static std::unique_ptr<juce::FileChooser> currentChooser;
    currentChooser = std::move(chooser);
}

void PluginEditor::updateFromProcessor() {
    // Sync playback state
    bool playing = processor.isPlaying.load();
    float progress = processor.playbackProgress.load();
    
    transportBar.setPlaying(playing);
    transportBar.setProgress(progress);
    waveformDisplay.setPlaying(playing);
    waveformDisplay.setProgress(progress);
}

} // namespace kelly
