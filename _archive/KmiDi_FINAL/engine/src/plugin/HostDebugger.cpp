#include "plugin/HostDebugger.h"
#include <algorithm>

namespace kelly {

HostDebugger::HostDebugger() : currentHost_(HostType::Unknown) {
}

HostType HostDebugger::detectHostType(juce::AudioProcessor* processor) {
    if (!processor) {
        return HostType::Unknown;
    }
    
    // Try to detect from processor name or wrapper
    juce::String name = processor->getName();
    juce::String wrapper = juce::PluginHostType::getPluginLoadedAs();
    
    if (wrapper.containsIgnoreCase("REAPER")) {
        return HostType::Reaper;
    } else if (wrapper.containsIgnoreCase("Logic") || wrapper.containsIgnoreCase("AU")) {
        return HostType::LogicPro;
    } else if (wrapper.containsIgnoreCase("Ableton") || wrapper.containsIgnoreCase("Live")) {
        return HostType::AbletonLive;
    } else if (wrapper.containsIgnoreCase("Pro Tools") || wrapper.containsIgnoreCase("AAX")) {
        return HostType::ProTools;
    } else if (wrapper.containsIgnoreCase("Cubase")) {
        return HostType::Cubase;
    } else if (wrapper.containsIgnoreCase("Studio One")) {
        return HostType::StudioOne;
    } else if (wrapper.containsIgnoreCase("FL Studio")) {
        return HostType::FLStudio;
    }
    
    return HostType::Unknown;
}

std::string HostDebugger::getHostName(HostType host) {
    switch (host) {
        case HostType::Reaper:
            return "Reaper";
        case HostType::LogicPro:
            return "Logic Pro";
        case HostType::AbletonLive:
            return "Ableton Live";
        case HostType::ProTools:
            return "Pro Tools";
        case HostType::Cubase:
            return "Cubase";
        case HostType::StudioOne:
            return "Studio One";
        case HostType::FLStudio:
            return "FL Studio";
        default:
            return "Unknown";
    }
}

bool HostDebugger::hasQuirks(HostType host) const {
    // Known hosts with quirks
    return host == HostType::LogicPro || 
           host == HostType::AbletonLive || 
           host == HostType::ProTools;
}

void HostDebugger::logQuirk(const std::string& quirk) {
    if (std::find(loggedQuirks_.begin(), loggedQuirks_.end(), quirk) == loggedQuirks_.end()) {
        loggedQuirks_.push_back(quirk);
    }
}

bool HostDebugger::verifyParameterSync(
    HostType host,
    const std::string& parameter_id,
    float expected_value,
    float actual_value
) const {
    // Host-specific tolerance
    float tolerance = 0.001f; // Default
    
    switch (host) {
        case HostType::ProTools:
            tolerance = 0.01f; // Pro Tools has lower precision
            break;
        case HostType::AbletonLive:
            tolerance = 0.0001f; // Ableton is very precise
            break;
        default:
            break;
    }
    
    return std::abs(expected_value - actual_value) < tolerance;
}

std::string HostDebugger::getWorkaround(HostType host, const std::string& quirk) const {
    // Return known workarounds for host quirks
    if (host == HostType::LogicPro && quirk == "editor_recreation") {
        return "Save state before editor destruction";
    } else if (host == HostType::AbletonLive && quirk == "automation_override") {
        return "Ignore automation during UI drag";
    } else if (host == HostType::ProTools && quirk == "parameter_precision") {
        return "Use lower precision for parameter values";
    }
    
    return "No known workaround";
}

} // namespace kelly
