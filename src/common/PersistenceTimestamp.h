#pragma once

#include <cstdlib>
#include <optional>
#include <juce_core/juce_core.h>

namespace kelly::persistence {

inline juce::Time zeroTime() {
    return juce::Time(static_cast<juce::int64>(0));
}

inline std::optional<juce::Time> parseSourceDateEpoch() {
    const char* raw = std::getenv("SOURCE_DATE_EPOCH");
    if (raw == nullptr || *raw == '\0') {
        return std::nullopt;
    }

    char* end = nullptr;
    const auto seconds = std::strtoll(raw, &end, 10);
    if (end == raw || (end != nullptr && *end != '\0') || seconds < 0) {
        return std::nullopt;
    }

    return juce::Time(static_cast<juce::int64>(seconds) * 1000);
}

inline juce::Time resolvePersistenceTimestamp() {
    if (const auto deterministic = parseSourceDateEpoch(); deterministic.has_value()) {
        return *deterministic;
    }
    return juce::Time::getCurrentTime();
}

} // namespace kelly::persistence
