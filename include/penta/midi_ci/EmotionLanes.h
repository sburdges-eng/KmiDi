#pragma once
/**
 * EmotionLanes.h — MIDI-CI Property Exchange for Valence/Arousal lanes.
 *
 * Exposes two high-resolution emotion parameters over MIDI:
 *   - Valence [-1.0, +1.0] — affective pleasantness
 *   - Arousal [ 0.0,  1.0] — energy/activation
 *
 * Transport:
 *   MIDI-1.0: 14-bit CC pairs (Valence CC 20/52, Arousal CC 21/53) on Ch 1
 *   MIDI-2.0: UMP Assignable Parameter (index 0/1, Group 0, Ch 0)
 *
 * RT-safe: all conversion functions are constexpr/inline, no allocations.
 */

#include <cstdint>
#include <cmath>
#include <array>

namespace penta {
namespace midi_ci {

// -------------------------------------------------------------------------
// Emotion lane definitions
// -------------------------------------------------------------------------

struct EmotionLane {
    const char* id;
    const char* label;
    float minVal;
    float maxVal;
    float defaultVal;
    uint8_t midi1MsbCC;
    uint8_t midi1LsbCC;
    uint8_t midi1Channel;  // 0-based
    uint8_t midi2Index;    // UMP assignable parameter index
};

static constexpr EmotionLane kValenceLane = {
    "valence", "Valence", -1.0f, 1.0f, 0.0f,
    20, 52, 0, 0
};

static constexpr EmotionLane kArousalLane = {
    "arousal", "Arousal", 0.0f, 1.0f, 0.3f,
    21, 53, 0, 1
};

static constexpr std::array<EmotionLane, 2> kEmotionLanes = {kValenceLane, kArousalLane};

// -------------------------------------------------------------------------
// Scalar ↔ MIDI-1.0 CC14 (0..16383)
// -------------------------------------------------------------------------

inline uint16_t scalarToCC14(float x, float xmin, float xmax) {
    float t = (x - xmin) / (xmax - xmin);
    t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
    return static_cast<uint16_t>(std::lround(t * 16383.0f));
}

inline float cc14ToScalar(uint16_t v, float xmin, float xmax) {
    float t = static_cast<float>(v) / 16383.0f;
    return xmin + t * (xmax - xmin);
}

inline uint8_t cc14Msb(uint16_t v) { return (v >> 7) & 0x7F; }
inline uint8_t cc14Lsb(uint16_t v) { return v & 0x7F; }
inline uint16_t cc14FromBytes(uint8_t msb, uint8_t lsb) {
    return static_cast<uint16_t>((msb & 0x7F) << 7) | (lsb & 0x7F);
}

// -------------------------------------------------------------------------
// Scalar ↔ MIDI-2.0 32-bit (0..0xFFFFFFFF)
// -------------------------------------------------------------------------

inline uint32_t scalarToU32(float x, float xmin, float xmax) {
    double t = static_cast<double>(x - xmin) / static_cast<double>(xmax - xmin);
    t = t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
    return static_cast<uint32_t>(std::llround(t * 4294967295.0));
}

inline float u32ToScalar(uint32_t v, float xmin, float xmax) {
    double t = static_cast<double>(v) / 4294967295.0;
    return static_cast<float>(xmin + t * (xmax - xmin));
}

// -------------------------------------------------------------------------
// Convenience: emit valence/arousal as CC14 bytes
// -------------------------------------------------------------------------

struct CC14Pair {
    uint8_t channel;
    uint8_t msbCC;
    uint8_t msbVal;
    uint8_t lsbCC;
    uint8_t lsbVal;
};

inline CC14Pair valenceToCC14(float valence) {
    uint16_t v = scalarToCC14(valence, kValenceLane.minVal, kValenceLane.maxVal);
    return {kValenceLane.midi1Channel, kValenceLane.midi1MsbCC, cc14Msb(v),
            kValenceLane.midi1LsbCC, cc14Lsb(v)};
}

inline CC14Pair arousalToCC14(float arousal) {
    uint16_t v = scalarToCC14(arousal, kArousalLane.minVal, kArousalLane.maxVal);
    return {kArousalLane.midi1Channel, kArousalLane.midi1MsbCC, cc14Msb(v),
            kArousalLane.midi1LsbCC, cc14Lsb(v)};
}

// -------------------------------------------------------------------------
// Property Exchange JSON (compile-time string for PE responder)
// -------------------------------------------------------------------------

inline const char* getEmotionLanesPEJson() {
    return R"({
  "pe_version": "1.0",
  "provider": "KmiDi",
  "resource": "emotion_lanes",
  "schema": "com.kmidi/emotion.v1",
  "transport": ["midi1", "midi2"],
  "properties": [
    {
      "id": "valence",
      "label": "Valence",
      "desc": "Affective pleasantness",
      "range": { "min": -1.0, "max": 1.0, "default": 0.0 },
      "midi": {
        "midi1": { "mode": "cc14", "msb_cc": 20, "lsb_cc": 52, "channel": 1 },
        "midi2": { "group": 0, "channel": 0, "attr": "assignable-parameter", "index": 0 }
      }
    },
    {
      "id": "arousal",
      "label": "Arousal",
      "desc": "Energy/activation",
      "range": { "min": 0.0, "max": 1.0, "default": 0.3 },
      "midi": {
        "midi1": { "mode": "cc14", "msb_cc": 21, "lsb_cc": 53, "channel": 1 },
        "midi2": { "group": 0, "channel": 0, "attr": "assignable-parameter", "index": 1 }
      }
    }
  ]
})";
}

} // namespace midi_ci
} // namespace penta
