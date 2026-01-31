#include "ui/GenerateButton.h"
#include "ui/KellyLookAndFeel.h"

namespace kelly {

GenerateButton::GenerateButton() : juce::TextButton("GENERATE") {
    setColour(juce::TextButton::buttonColourId, juce::Colour(0xFF4A90E2));
    setColour(juce::TextButton::buttonOnColourId, juce::Colour(0xFF357ABD));
    setColour(juce::TextButton::textColourOffId, juce::Colours::white);
    setColour(juce::TextButton::textColourOnId, juce::Colours::white);
}

void GenerateButton::paintButton(juce::Graphics& g, bool shouldDrawButtonAsHighlighted,
                                  bool shouldDrawButtonAsDown) {
    auto bounds = getLocalBounds().toFloat();
    
    // Draw button background with rounded corners
    juce::Colour baseColour = findColour(juce::TextButton::buttonColourId);
    if (shouldDrawButtonAsDown || isDown()) {
        baseColour = findColour(juce::TextButton::buttonOnColourId);
    } else if (shouldDrawButtonAsHighlighted || isOver()) {
        baseColour = baseColour.brighter(0.2f);
    }
    
    g.setColour(baseColour);
    g.fillRoundedRectangle(bounds, 6.0f);
    
    // Draw border
    g.setColour(baseColour.darker(0.3f));
    g.drawRoundedRectangle(bounds, 6.0f, 2.0f);
    
    // Draw text
    g.setColour(findColour(juce::TextButton::textColourOffId));
    g.setFont(16.0f);
    g.drawText(getButtonText(), bounds, juce::Justification::centred);
}

void GenerateButton::startGenerateAnimation() {
    isAnimating_ = true;
    animationProgress_ = 0.0f;
}

void GenerateButton::stopGenerateAnimation() {
    isAnimating_ = false;
}

} // namespace kelly
