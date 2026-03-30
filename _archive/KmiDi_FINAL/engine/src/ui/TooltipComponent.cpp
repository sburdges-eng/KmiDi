#include "TooltipComponent.h"
#include "KellyLookAndFeel.h"

namespace kelly {

namespace {
TooltipComponent& getSharedTooltip() {
    static TooltipComponent tooltip;
    static bool initialized = false;
    if (!initialized) {
        tooltip.addToDesktop(juce::ComponentPeer::windowIsTemporary);
        tooltip.setVisible(false);
        initialized = true;
    }
    return tooltip;
}
} // namespace

TooltipComponent::TooltipComponent() {
    setOpaque(false);
    setAlwaysOnTop(true);
    setInterceptsMouseClicks(false, false);
}

void TooltipComponent::showTooltip(juce::Component* target, const juce::String& text, int timeoutMs) {
    if (text.isEmpty()) {
        return;
    }

    auto& tooltip = getSharedTooltip();
    tooltip.tooltipText_ = text;

    const juce::Font font(11.0f);
    const int textWidth = font.getStringWidth(text);
    const int padding = 12;
    const int width = textWidth + padding * 2;
    const int height = static_cast<int>(font.getHeight()) + padding;

    juce::Rectangle<int> targetBounds;
    if (target != nullptr) {
        targetBounds = target->getScreenBounds();
    } else {
        targetBounds = juce::Desktop::getInstance().getDisplays().getPrimaryDisplay()->totalArea;
    }

    const int x = targetBounds.getX() + (targetBounds.getWidth() - width) / 2;
    const int y = targetBounds.getY() - height - 6;
    tooltip.setBounds(x, y, width, height);
    tooltip.setVisible(true);
    tooltip.repaint();

    if (timeoutMs > 0) {
        juce::Timer::callAfterDelay(timeoutMs, [] { TooltipComponent::hideTooltip(); });
    }
}

void TooltipComponent::hideTooltip() {
    auto& tooltip = getSharedTooltip();
    tooltip.tooltipText_.clear();
    tooltip.setVisible(false);
}

void TooltipComponent::paint(juce::Graphics& g) {
    auto bounds = getLocalBounds().toFloat();

    // Modern tooltip background
    g.setColour(KellyLookAndFeel::surfaceColor.withAlpha(0.95f));
    g.fillRoundedRectangle(bounds, 6.0f);

    // Border
    g.setColour(KellyLookAndFeel::borderColor);
    g.drawRoundedRectangle(bounds, 6.0f, 1.0f);

    // Text
    g.setColour(KellyLookAndFeel::textPrimary);
    g.setFont(juce::Font(11.0f));
    g.drawText(tooltipText_, bounds.reduced(8.0f), juce::Justification::centredLeft, true);
}

void TooltipComponent::resized() {
    // Auto-sized based on text
}

} // namespace kelly
