#include "SidePanel.h"
#include "KellyLookAndFeel.h"

namespace kelly {

SidePanel::SidePanel(Side side) : side_(side) {
    setOpaque(true);

    juce::String labelText = (side == Side::SideA) ? "Side A" : "Side B";
    label_.setText(labelText, juce::dontSendNotification);
    label_.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(label_);

    const juce::String placeholder = (side == Side::SideA)
        ? "Where you are..."
        : "Where you want to go...";
    input_.setMultiLine(true);
    input_.setReturnKeyStartsNewLine(true);
    input_.setTextToShowWhenEmpty(placeholder, juce::Colours::grey);
    addAndMakeVisible(input_);

    intensity_.setRange(0.0, 1.0, 0.01);
    intensity_.setValue(0.5);
    intensity_.setSliderStyle(juce::Slider::LinearHorizontal);
    intensity_.setTextBoxStyle(juce::Slider::TextBoxRight, false, 60, 20);
    addAndMakeVisible(intensity_);
}

void SidePanel::paint(juce::Graphics& g) {
    auto bounds = getLocalBounds();

    // Draw panel background
    juce::Colour bgColour = (side_ == Side::SideA)
        ? KellyLookAndFeel::primaryColor.withAlpha(0.2f)
        : KellyLookAndFeel::secondaryColor.withAlpha(0.2f);

    g.fillAll(bgColour);

    // Draw border
    g.setColour(bgColour.darker(0.2f));
    g.drawRect(bounds, 1);

    // Label is drawn by the label component
}

void SidePanel::resized() {
    // Layout label
    auto labelArea = getLocalBounds().removeFromTop(25);
    label_.setBounds(labelArea);

    auto contentArea = getLocalBounds().reduced(8);
    auto sliderArea = contentArea.removeFromBottom(40);
    intensity_.setBounds(sliderArea);
    input_.setBounds(contentArea);
}

SideA SidePanel::getSideAState() const {
    if (side_ == Side::SideA) {
        return SideA{input_.getText().toStdString(),
                     static_cast<float>(intensity_.getValue()),
                     std::nullopt};
    }
    return SideA{"", 0.5f, std::nullopt};
}

SideB SidePanel::getSideBState() const {
    if (side_ == Side::SideB) {
        return SideB{input_.getText().toStdString(),
                     static_cast<float>(intensity_.getValue()),
                     std::nullopt};
    }
    return SideB{"", 0.5f, std::nullopt};
}

} // namespace kelly
