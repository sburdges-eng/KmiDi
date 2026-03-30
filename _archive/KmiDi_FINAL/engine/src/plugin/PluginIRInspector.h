#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include "core/intent_ir/IntentFrame.h"
#include <atomic>
#include <mutex>

namespace kelly {

/**
 * PluginIRInspector - JUCE component displaying IntentFrame
 * 
 * Features:
 * - Read-only display
 * - Provenance visualization
 * - Thread-safe updates
 */
class PluginIRInspector : public juce::Component {
public:
    PluginIRInspector();
    ~PluginIRInspector() override = default;
    
    void paint(juce::Graphics& g) override;
    void resized() override;
    
    /**
     * Update IntentFrame (thread-safe)
     * Can be called from any thread
     */
    void updateFrame(const intent_ir::IntentFrame& frame);
    
    /**
     * Clear display
     */
    void clear();
    
private:
    // Thread-safe frame storage
    std::mutex frameMutex_;
    intent_ir::IntentFrame currentFrame_;
    std::atomic<bool> hasFrame_{false};
    
    // UI components
    juce::Label titleLabel_;
    juce::TextEditor frameDisplay_;
    
    // Helper methods
    void updateDisplay();
    juce::String formatFrame(const intent_ir::IntentFrame& frame);
    juce::Colour getProvenanceColor(intent_ir::IntentSource source);
    juce::String getProvenanceName(intent_ir::IntentSource source);
};

} // namespace kelly
