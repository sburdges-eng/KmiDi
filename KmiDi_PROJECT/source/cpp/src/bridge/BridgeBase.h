#pragma once

/**
 * BridgeBase.h - Base class for all bridge implementations
 * =======================================================
 *
 * Provides common interface and functionality for all bridges:
 * - Lifecycle management (initialize/shutdown)
 * - Availability checking
 * - Error logging
 * - Common state management
 */

#include <atomic>
#include <string>
#include <memory>

namespace kelly {
namespace bridge {

/**
 * BridgeBase - Abstract base class for all bridge implementations
 *
 * All bridges should inherit from this class to ensure consistent
 * interface and behavior across the bridge system.
 */
class BridgeBase {
public:
    BridgeBase(const std::string& bridgeName);
    virtual ~BridgeBase() = default;

    /**
     * Initialize the bridge
     * @return true if initialization successful
     */
    virtual bool initialize() = 0;

    /**
     * Shutdown the bridge and cleanup resources
     */
    virtual void shutdown() = 0;

    /**
     * Check if bridge is available and ready to use
     */
    bool isAvailable() const { return available_.load(); }

    /**
     * Get the bridge name (for logging/debugging)
     */
    const std::string& getName() const { return bridgeName_; }

protected:
    /**
     * Set availability status (should be called by derived classes)
     */
    void setAvailable(bool available) { available_.store(available); }

    /**
     * Log an error message (can be overridden for custom logging)
     */
    virtual void logError(const std::string& message) const;
    virtual void logInfo(const std::string& message) const;

private:
    std::atomic<bool> available_{false};
    std::string bridgeName_;
};

} // namespace bridge
} // namespace kelly
