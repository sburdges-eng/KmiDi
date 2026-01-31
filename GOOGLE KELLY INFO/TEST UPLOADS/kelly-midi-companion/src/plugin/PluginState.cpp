#include "plugin/PluginState.h"

namespace kelly {

void PluginState::saveEmotionSettings(const CassetteState& state) {
    // Placeholder - currently handled by JUCE's ValueTreeState
}

CassetteState PluginState::loadEmotionSettings() const {
    // Placeholder - currently handled by JUCE's ValueTreeState
    return CassetteState{
        SideA{"", 0.5f, std::nullopt},
        SideB{"", 0.5f, std::nullopt},
        false
    };
}

} // namespace kelly
