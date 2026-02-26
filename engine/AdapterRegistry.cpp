#include "engine/AdapterRegistry.h"

#include "adapters/legacy/EmotionThesaurusEnricher.h"

namespace kelly {

AdapterRegistry& AdapterRegistry::instance() {
    static AdapterRegistry registry;
    return registry;
}

AdapterRegistry::AdapterRegistry() {
    intentEnrichers_.push_back(&noOpIntentEnricher_);
    ownedIntentEnrichers_.push_back(std::make_unique<EmotionThesaurusEnricher>());

    for (const auto& enricher : ownedIntentEnrichers_) {
        intentEnrichers_.push_back(enricher.get());
    }
}

void AdapterRegistry::applyIntentEnrichers(const UserInput& input, IntentResult& intent) {
    const auto initialStatus = intent.status;
    for (auto* enricher : intentEnrichers_) {
        if (enricher != nullptr) {
            enricher->enrich(input, intent);
            // Intent enrichment is read-only for control flow: adapters cannot override status.
            intent.status = initialStatus;
        }
    }
}

std::vector<std::string> AdapterRegistry::activeIntentEnricherIds() const {
    std::vector<std::string> ids;
    ids.reserve(intentEnrichers_.size());
    for (auto* enricher : intentEnrichers_) {
        if (enricher != nullptr && enricher->id() != nullptr) {
            ids.emplace_back(enricher->id());
        }
    }
    return ids;
}

} // namespace kelly
