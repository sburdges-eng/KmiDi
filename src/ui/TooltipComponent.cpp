#include "TooltipComponent.h"
#include "KellyLookAndFeel.h"

namespace kelly {

TooltipComponent* TooltipComponent::sharedInstance_ = nullptr;

TooltipComponent* TooltipComponent::getOrCreateShared() {
    // Heap-allocated so juce::DeletedAtShutdown arranges the delete via
    // juce::MessageManager teardown. The destructor clears sharedInstance_
    // so any pending callAfterDelay lambda that fires after shutdown can
    // null-check the pointer instead of dereferencing freed memory.
    if (sharedInstance_ == nullptr) {
        sharedInstance_ = new TooltipComponent();
        sharedInstance_->addToDesktop(juce::ComponentPeer::windowIsTemporary);
        sharedInstance_->setVisible(false);
    }
    return sharedInstance_;
}

TooltipComponent::TooltipComponent() {
    setOpaque(false);
    setAlwaysOnTop(true);
    setInterceptsMouseClicks(false, false);
}

TooltipComponent::~TooltipComponent() {
    if (sharedInstance_ == this) {
        sharedInstance_ = nullptr;
    }
}

void TooltipComponent::showTooltip(juce::Component* target, const juce::String& text, int timeoutMs) {
    if (text.isEmpty()) {
        return;
    }

    auto* tip = getOrCreateShared();
    if (tip == nullptr) {
        return;
    }
    auto& tooltip = *tip;
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
    // After juce::DeletedAtShutdown reaps the singleton (e.g. on app
    // teardown) any pending callAfterDelay lambda that lands here would
    // otherwise UAF the freed instance — null-check, do not recreate.
    if (sharedInstance_ == nullptr) {
        return;
    }
    sharedInstance_->tooltipText_.clear();
    sharedInstance_->setVisible(false);
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
