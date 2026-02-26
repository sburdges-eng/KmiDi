#pragma once

#include "common/Types.h"
#include <optional>
#include <vector>

namespace kelly {

class IHarmonyEngine {
public:
    virtual ~IHarmonyEngine() = default;
    virtual const char* id() const = 0;
    virtual std::optional<std::vector<Chord>> generateChords(const IntentResult& intent) = 0;
};

} // namespace kelly
