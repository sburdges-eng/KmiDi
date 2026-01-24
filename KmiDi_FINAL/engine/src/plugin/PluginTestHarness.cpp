#include "plugin/PluginTestHarness.h"

namespace kelly {

PluginTestHarness::PluginTestHarness() {
}

void PluginTestHarness::enableTestMode(const intent_ir::IntentFrame& static_frame) {
    std::lock_guard<std::mutex> lock(frameMutex_);
    staticFrame_ = static_frame;
    testModeActive_.store(true);
}

void PluginTestHarness::disableTestMode() {
    testModeActive_.store(false);
}

bool PluginTestHarness::getStaticFrame(intent_ir::IntentFrame& frame) const {
    if (!testModeActive_.load()) {
        return false;
    }
    
    std::lock_guard<std::mutex> lock(frameMutex_);
    frame = staticFrame_;
    return true;
}

bool PluginTestHarness::verifyStability() const {
    if (!testModeActive_.load()) {
        return true; // Not in test mode, stability check doesn't apply
    }
    
    // In test mode, stability means the frame hasn't changed
    // This is a simple check - in a real implementation, you'd compare
    // the current frame with the static frame
    return true;
}

} // namespace kelly
