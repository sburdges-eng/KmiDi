#include "ui/KellyLookAndFeel.h"

namespace kelly {

const juce::Colour KellyLookAndFeel::cassetteBrown(0xFF8B7355);
const juce::Colour KellyLookAndFeel::cassetteLabel(0xFFF5F5DC);
const juce::Colour KellyLookAndFeel::sideAHighlight(0xFFE8E8E8);
const juce::Colour KellyLookAndFeel::sideBHighlight(0xFFD3D3D3);

KellyLookAndFeel::KellyLookAndFeel() {
    // Set default colors for cassette aesthetic
    setColour(juce::ResizableWindow::backgroundColourId, cassetteBrown);
    setColour(juce::TextButton::buttonColourId, juce::Colour(0xFF4A90E2));
    setColour(juce::TextButton::textColourOffId, juce::Colours::white);
}

void KellyLookAndFeel::drawButtonBackground(juce::Graphics& g, juce::Button& button,
                                            const juce::Colour& backgroundColour,
                                            bool shouldDrawButtonAsHighlighted,
                                            bool shouldDrawButtonAsDown) {
    auto bounds = button.getLocalBounds().toFloat();
    juce::Colour baseColour = backgroundColour;
    
    if (shouldDrawButtonAsDown) {
        baseColour = baseColour.darker(0.2f);
    } else if (shouldDrawButtonAsHighlighted) {
        baseColour = baseColour.brighter(0.1f);
    }
    
    g.setColour(baseColour);
    g.fillRoundedRectangle(bounds, 4.0f);
    
    g.setColour(baseColour.darker(0.3f));
    g.drawRoundedRectangle(bounds, 4.0f, 1.0f);
}

void KellyLookAndFeel::drawTextEditorOutline(juce::Graphics& g, int width, int height,
                                              juce::TextEditor& textEditor) {
    if (textEditor.isEnabled()) {
        g.setColour(textEditor.findColour(juce::TextEditor::outlineColourId));
        g.drawRect(0, 0, width, height, 1);
    }
}

} // namespace kelly
