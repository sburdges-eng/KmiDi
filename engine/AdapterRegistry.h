#pragma once

#include "adapters/legacy/NoOpAdapters.h"
#include "common/Types.h"
#include <memory>
#include <string>
#include <vector>

namespace kelly {

class IIntentEnricher;

class AdapterRegistry {
public:
    static AdapterRegistry& instance();

    void applyIntentEnrichers(const UserInput& input, IntentResult& intent);
    std::vector<std::string> activeIntentEnricherIds() const;

private:
    AdapterRegistry();

    // Default no-op adapters keep existing behavior stable while wiring proceeds.
    NoOpIntentEnricher noOpIntentEnricher_;
    NoOpHarmonyEngine noOpHarmonyEngine_;
    NoOpGrooveEngine noOpGrooveEngine_;
    NoOpArrangementEngine noOpArrangementEngine_;
    NoOpMLAssistService noOpMLAssistService_;
    NoOpVoiceLayer noOpVoiceLayer_;
    NoOpBiometricInput noOpBiometricInput_;

    std::vector<IIntentEnricher*> intentEnrichers_;
    std::vector<std::unique_ptr<IIntentEnricher>> ownedIntentEnrichers_;
};

} // namespace kelly
