#pragma once

namespace kelly {

class IVoiceLayer {
public:
    virtual ~IVoiceLayer() = default;
    virtual const char* id() const = 0;
    virtual bool isAvailable() const = 0;
    virtual void reset() = 0;
};

} // namespace kelly
