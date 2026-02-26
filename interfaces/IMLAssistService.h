#pragma once

#include "common/Types.h"
#include <optional>

namespace kelly {

class IMLAssistService {
public:
    virtual ~IMLAssistService() = default;
    virtual const char* id() const = 0;
    virtual std::optional<IntentResult> inferBias(const UserInput& input) = 0;
};

} // namespace kelly
