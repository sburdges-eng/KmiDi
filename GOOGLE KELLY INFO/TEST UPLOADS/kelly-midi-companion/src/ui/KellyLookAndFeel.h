#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

namespace kelly {

/**
 * Kelly Look and Feel - Custom styling for the cassette aesthetic.
 * 
 * Currently using default JUCE look and feel.
 * This will be implemented in Phase 3 for visual polish.
 */
class KellyLookAndFeel : public juce::LookAndFeel_V4 {
public:
    KellyLookAndFeel();
    ~KellyLookAndFeel() override = default;
    
    // Custom colors for cassette theme
    static const juce::Colour cassetteBrown;
    static const juce::Colour cassetteLabel;
    static const juce::Colour sideAHighlight;
    static const juce::Colour sideBHighlight;
    
    // Override drawing methods for custom styling
    void drawButtonBackground(juce::Graphics& g, juce::Button& button,
                              const juce::Colour& backgroundColour,
                              bool shouldDrawButtonAsHighlighted,
                              bool shouldDrawButtonAsDown) override;
    
    void drawTextEditorOutline(juce::Graphics& g, int width, int height,
                               juce::TextEditor& textEditor) override;
    
private:
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(KellyLookAndFeel)
};

} // namespace kelly

