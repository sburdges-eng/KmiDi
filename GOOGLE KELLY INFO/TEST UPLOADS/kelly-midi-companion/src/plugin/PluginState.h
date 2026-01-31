#pragma once

#include "common/Types.h"
#include <juce_audio_processors/juce_audio_processors.h>

namespace kelly {

/**
 * Plugin State - Manages plugin state persistence.
 * 
 * Currently handled by JUCE's AudioProcessorValueTreeState.
 * This is a placeholder for custom state management if needed.
 */
class PluginState {
public:
    PluginState() = default;
    ~PluginState() = default;
    
    // Placeholder for future custom state management
    // Currently using JUCE's built-in state system
    
    /**
     * Save emotion settings to state.
     */
    void saveEmotionSettings(const CassetteState& state);
    
    /**
     * Load emotion settings from state.
     */
    CassetteState loadEmotionSettings() const;
};

} // namespace kelly

