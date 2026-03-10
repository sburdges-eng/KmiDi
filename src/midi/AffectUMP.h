#pragma once

#include <cstdint>

namespace kelly {

/** KmiDi MIDI 2.0 affect channel: vendor CC indices and UMP helpers. See midi/README.md. */
namespace AffectUMP {

/** Vendor assignable controller indices (UMP Channel Voice 32-bit CC). */
constexpr uint8_t kCCValence = 0x28;
constexpr uint8_t kCCArousal = 0x29;
constexpr uint8_t kCCDynamics = 0x2A;

/** Target control rate (Hz) for affect lanes (100–250 Hz). */
constexpr int kControlRateHz = 100;

/**
 * Map a float in [min, max] to 32-bit UMP value [0, 0xFFFFFFFF].
 * Clamps value to [min, max] then linear scale.
 */
uint32_t floatToUmp32Value(float value, float minVal, float maxVal);

/**
 * Build one MIDI 2.0 Channel Voice Control Change (32-bit) packet for affect.
 * Uses juce::universal_midi_packets::Factory::makeControlChangeV2.
 * group: UMP group (usually 0), channel: 0..15, indexCC: 0x28/0x29/0x2A, value: float in range.
 * Valence: use min=-1, max=1; arousal/dynamics: min=0, max=1.
 * Returns a 2-word packet; copy to UMP output or MidiBuffer (as UMP).
 */
void buildUmpAffectCC(uint8_t group,
                      uint8_t channel,
                      uint8_t indexCC,
                      float value,
                      float minVal,
                      float maxVal,
                      uint32_t outWords[2]);

}  // namespace AffectUMP
}  // namespace kelly
