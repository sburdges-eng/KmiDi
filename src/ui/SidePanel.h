#pragma once
/*
 * SidePanel.h - Cassette Side A / Side B Input Strip
 * ===================================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - UI Layer: CassetteView, PluginEditor (Side A “where you are”, Side B goals)
 * - Common Layer: Types.h (SideA / SideB state structs)
 * - Engine Layer: Intent and emotion capture from UI
 *
 * Purpose: Captures paired emotional/theory states for generative contrast.
 *
 * Features:
 * - Distinct paint/layout per Side enum
 * - State getters for host logic
 */

#include <juce_gui_basics/juce_gui_basics.h>
#include "../common/Types.h"
#include <optional>

namespace kelly {

class SidePanel : public juce::Component {
public:
    enum class Side {
        SideA,  // "Where you are"
        SideB   // "Where you want to go"
    };
    
    explicit SidePanel(Side side);
    ~SidePanel() override = default;
    
    void paint(juce::Graphics& g) override;
    void resized() override;
    
    // Get current state
    SideA getSideAState() const;
    SideB getSideBState() const;
    
    // Access to components for external setup
    juce::TextEditor& getInputEditor() { return input_; }
    juce::Slider& getIntensitySlider() { return intensity_; }
    
private:
    Side side_;
    juce::TextEditor input_;
    juce::Slider intensity_;
    juce::Label label_;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SidePanel)
};

} // namespace kelly

