#pragma once

#include <optional>

namespace kelly {

struct BiometricSnapshot {
    std::optional<float> heartRateBpm;
    std::optional<float> stressLevel;
    std::optional<float> respirationRate;
};

class IBiometricInput {
public:
    virtual ~IBiometricInput() = default;
    virtual const char* id() const = 0;
    virtual std::optional<BiometricSnapshot> readSnapshot() = 0;
};

} // namespace kelly
