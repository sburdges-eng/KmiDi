#pragma once

#include "common/Types.h"

namespace kelly {

class IArrangementEngine {
public:
    virtual ~IArrangementEngine() = default;
    virtual const char* id() const = 0;
    virtual void arrange(const IntentResult& intent, GeneratedMidi& midi) = 0;
};

} // namespace kelly
