#pragma once
/*
 * TooltipComponent.h - Transient Tooltip Overlay
 * ===============================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - UI Layer: EmotionWorkstation, PluginEditor (attach via TooltipHelper)
 * - Framework: juce::Component help text integration
 *
 * Purpose: Lightweight overlay and helpers for discoverability copy.
 *
 * Features:
 * - Timed show/hide singleton-style API
 * - TooltipHelper::setTooltip wraps setHelpText
 */

#include <juce_gui_basics/juce_gui_basics.h>

namespace kelly {

class TooltipComponent : public juce::Component, public juce::DeletedAtShutdown {
public:
    TooltipComponent();
    ~TooltipComponent() override;

    static void showTooltip(juce::Component* target, const juce::String& text, int timeoutMs = 3000);
    static void hideTooltip();

    void paint(juce::Graphics& g) override;
    void resized() override;

private:
    juce::String tooltipText_;
    juce::Point<int> targetPosition_;

    // The singleton instance is heap-allocated via getOrCreateShared() and
    // freed by juce::DeletedAtShutdown at MessageManager teardown. The dtor
    // clears this pointer so any pending callAfterDelay lambda that fires
    // during shutdown can null-check before dereferencing.
    static TooltipComponent* sharedInstance_;
    static TooltipComponent* getOrCreateShared();

    JUCE_DECLARE_NON_COPYABLE(TooltipComponent)
};

/**
 * Helper class to add tooltips to components
 */
class TooltipHelper {
public:
    static void setTooltip(juce::Component* component, const juce::String& tooltip) {
        if (component) {
            component->setHelpText(tooltip);
        }
    }
};

} // namespace kelly
