#pragma once
/*
 * CassetteView.h - Cassette Chrome and Content Host
 * ==================================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - UI Layer: SidePanel, EmotionWorkstation (nested content)
 * - UI Layer: KellyLookAndFeel (shared colors for chrome)
 *
 * Purpose: Animated cassette metaphor wrapping the main editor surface.
 *
 * Features:
 * - Reel animation timer
 * - Optional embedded content component
 * - Label and window customization
 */

#include <juce_gui_basics/juce_gui_basics.h>

namespace kelly {

class CassetteView : public juce::Component,
                     public juce::Timer {
public:
    CassetteView();
    ~CassetteView() override;
    
    void paint(juce::Graphics& g) override;
    void resized() override;
    void timerCallback() override;
    
    /** Set the content component to display inside the cassette */
    void setContentComponent(juce::Component* component);
    
    /** Set label text */
    void setLabelText(const juce::String& text);
    
    /** Start/stop tape animation */
    void setTapeAnimating(bool animating);
    
    /** Set tape position (0.0 to 1.0) */
    void setTapePosition(float position);
    
private:
    juce::Component* contentComponent_ = nullptr;
    juce::String labelText_ = "KELLY MIDI COMPANION";
    bool isAnimating_ = false;
    float tapePosition_ = 0.0f;
    float animationPhase_ = 0.0f;
    
    void drawCassetteBody(juce::Graphics& g, const juce::Rectangle<int>& bounds);
    void drawTapeReels(juce::Graphics& g, const juce::Rectangle<int>& bounds);
    void drawTapeWindow(juce::Graphics& g, const juce::Rectangle<int>& bounds);
    void drawLabel(juce::Graphics& g, const juce::Rectangle<int>& bounds);
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(CassetteView)
};

} // namespace kelly

