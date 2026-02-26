#pragma once

#include "common/Types.h"

namespace kelly {

class IIntentEnricher {
public:
    virtual ~IIntentEnricher() = default;
    virtual const char* id() const = 0;
    virtual void enrich(const UserInput& input, IntentResult& intent) = 0;
};

} // namespace kelly
