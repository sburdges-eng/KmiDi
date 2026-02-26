#pragma once

#include "interfaces/IIntentEnricher.h"

namespace kelly {

class EmotionThesaurusEnricher final : public IIntentEnricher {
public:
    const char* id() const override { return "legacy.emotion_thesaurus_enricher"; }
    void enrich(const UserInput& input, IntentResult& intent) override;
};

} // namespace kelly
