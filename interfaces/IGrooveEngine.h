#pragma once

#include "common/Types.h"

namespace kelly {

class IGrooveEngine {
public:
    virtual ~IGrooveEngine() = default;
    virtual const char* id() const = 0;
    virtual void applyGroove(const IntentResult& intent, GeneratedMidi& midi) = 0;
};

} // namespace kelly
