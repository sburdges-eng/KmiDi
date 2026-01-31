#pragma once

#include <cstdint>
#include <string>
#include <optional>
#include <unordered_map>

namespace emidi {

/**
 * Canonical emotional MIDI event.
 * Mirrors the Python EMIDI schema so emotional metadata flows through C++ cores.
 */
struct Event {
    // Core MIDI attributes
    std::uint8_t note = 60;          ///< MIDI pitch (0-127)
    std::uint8_t velocity = 80;      ///< MIDI velocity (0-127)
    std::uint8_t channel = 0;        ///< MIDI channel (0-15)
    std::uint32_t start_tick = 0;    ///< Start time in ticks
    std::uint32_t duration_ticks = 480; ///< Duration in ticks
    std::uint32_t track = 0;         ///< Logical track/index

    // Emotional metadata
    std::string intent_id;           ///< Link back to CompleteSongIntent
    std::string rule_break;          ///< Applied rule (e.g., HARMONY_ModalInterchange)
    std::string emotional_effect;    ///< Human-readable effect ("yearning", "defiance")
    float vulnerability = 0.5f;      ///< 0.0-1.0 vulnerability knob
    float complexity = 0.5f;         ///< 0.0-1.0 complexity knob
    float swing = 0.0f;              ///< swing offset applied

    // Arbitrary extra metadata (effects, groove IDs, etc.)
    std::unordered_map<std::string, std::string> tags;
};

}  // namespace emidi

