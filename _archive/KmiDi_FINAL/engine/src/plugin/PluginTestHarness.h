#pragma once

#include "core/intent_ir/IntentFrame.h"
#include <atomic>
#include <memory>

namespace kelly {

/**
 * PluginTestHarness - Golden test for plugin stability
 * 
 * Features:
 * - Static IntentFrame injection
 * - Disable UI edits
 * - Disable ML
 * - Disable host automation
 * - Verify stability
 */
class PluginTestHarness {
public:
    PluginTestHarness();
    ~PluginTestHarness() = default;
    
    /**
     * Enable test mode with static IntentFrame
     */
    void enableTestMode(const intent_ir::IntentFrame& static_frame);
    
    /**
     * Disable test mode
     */
    void disableTestMode();
    
    /**
     * Check if test mode is active
     */
    bool isTestModeActive() const { return testModeActive_.load(); }
    
    /**
     * Get current static frame (if test mode is active)
     */
    bool getStaticFrame(intent_ir::IntentFrame& frame) const;
    
    /**
     * Check if UI edits should be ignored
     */
    bool shouldIgnoreUIEdits() const { return testModeActive_.load(); }
    
    /**
     * Check if ML should be disabled
     */
    bool shouldDisableML() const { return testModeActive_.load(); }
    
    /**
     * Check if host automation should be ignored
     */
    bool shouldIgnoreAutomation() const { return testModeActive_.load(); }
    
    /**
     * Verify stability (check if frame hasn't changed unexpectedly)
     */
    bool verifyStability() const;
    
private:
    std::atomic<bool> testModeActive_{false};
    intent_ir::IntentFrame staticFrame_;
    mutable std::mutex frameMutex_;
};

} // namespace kelly
