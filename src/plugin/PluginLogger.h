#pragma once

#include <juce_core/juce_core.h>
#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <chrono>

namespace kelly {

/**
 * PluginLogger - Thread-safe logging for plugin lifecycle and Intent IR
 * 
 * Logs:
 * - Host lifecycle events
 * - Parameter changes (source tracking)
 * - IR updates
 */
class PluginLogger {
public:
    struct LogEntry {
        std::chrono::system_clock::time_point timestamp;
        std::string thread_name;
        std::string event_type;
        uint64_t intent_id;
        std::string parameter_source; // "UI", "host", "preset"
        std::string message;
    };
    
    PluginLogger();
    ~PluginLogger() = default;
    
    /**
     * Log host lifecycle event
     */
    void logLifecycle(const std::string& event, const std::string& details = "");
    
    /**
     * Log parameter change
     */
    void logParameterChange(
        const std::string& parameter_id,
        float new_value,
        const std::string& source
    );
    
    /**
     * Log IR update
     */
    void logIRUpdate(uint64_t intent_id, const std::string& source);
    
    /**
     * Get all log entries
     */
    std::vector<LogEntry> getEntries() const;
    
    /**
     * Export logs to file
     */
    bool exportToFile(const juce::File& file) const;
    
    /**
     * Clear logs
     */
    void clear();
    
    /**
     * Enable/disable logging
     */
    void setEnabled(bool enabled) { enabled_.store(enabled); }
    bool isEnabled() const { return enabled_.load(); }
    
private:
    mutable std::mutex entriesMutex_;
    std::vector<LogEntry> entries_;
    std::atomic<bool> enabled_{true};
    static constexpr size_t MAX_ENTRIES = 1000;
    
    std::string getCurrentThreadName() const;
    void addEntry(const LogEntry& entry);
};

} // namespace kelly
