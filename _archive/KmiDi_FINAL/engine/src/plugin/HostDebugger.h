#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <string>

namespace kelly {

/**
 * Host type enumeration
 */
enum class HostType {
    Unknown,
    Reaper,
    LogicPro,
    AbletonLive,
    ProTools,
    Cubase,
    StudioOne,
    FLStudio,
};

/**
 * HostDebugger - Host-specific debugging and workarounds
 * 
 * Features:
 * - Detect host type
 * - Host-specific workarounds
 * - Log host quirks
 * - Parameter sync verification
 */
class HostDebugger {
public:
    HostDebugger();
    ~HostDebugger() = default;
    
    /**
     * Detect host type from processor
     */
    static HostType detectHostType(juce::AudioProcessor* processor);
    
    /**
     * Get host name as string
     */
    static std::string getHostName(HostType host);
    
    /**
     * Check if host has known quirks
     */
    bool hasQuirks(HostType host) const;
    
    /**
     * Log host-specific quirk
     */
    void logQuirk(const std::string& quirk);
    
    /**
     * Verify parameter sync (host-specific)
     */
    bool verifyParameterSync(
        HostType host,
        const std::string& parameter_id,
        float expected_value,
        float actual_value
    ) const;
    
    /**
     * Get workaround for host quirk
     */
    std::string getWorkaround(HostType host, const std::string& quirk) const;
    
private:
    HostType currentHost_;
    std::vector<std::string> loggedQuirks_;
};

} // namespace kelly
