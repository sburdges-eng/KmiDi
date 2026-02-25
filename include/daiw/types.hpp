/**
 * @file types.hpp
 * @brief Core type definitions for DAiW
 *
 * Defines fundamental types used throughout the DAiW C++ codebase.
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <limits>

namespace daiw {

// =============================================================================
// Fundamental Types
// =============================================================================

using SampleRate = uint32_t;
using BufferSize = uint32_t;
using TickCount = int64_t;
using Tick = TickCount;  // MIDI ticks (PPQ-based), alias for midi.hpp
using MidiNote = uint8_t;
using MidiVelocity = uint8_t;
using MidiChannel = uint8_t;

// =============================================================================
// Audio Types
// =============================================================================

/// Sample type for audio processing (32-bit float)
using Sample = float;

/// High precision sample for accumulation
using SampleHigh = double;

/// Default sample rate
constexpr SampleRate DEFAULT_SAMPLE_RATE = 48000;

/// Default buffer size
constexpr BufferSize DEFAULT_BUFFER_SIZE = 512;

/// PPQ (Pulses Per Quarter note) for MIDI timing
constexpr int DEFAULT_PPQ = 480;

// =============================================================================
// MIDI Constants
// =============================================================================

constexpr MidiNote MIDI_NOTE_MIN = 0;
constexpr MidiNote MIDI_NOTE_MAX = 127;
constexpr MidiVelocity MIDI_VELOCITY_MIN = 0;
constexpr MidiVelocity MIDI_VELOCITY_MAX = 127;
constexpr MidiChannel MIDI_CHANNEL_MIN = 0;
constexpr MidiChannel MIDI_CHANNEL_MAX = 15;

// =============================================================================
// MIDI Event Types (for daiw/midi.hpp)
// =============================================================================

enum class MidiMessageType : uint8_t {
    NoteOff = 0x80,
    NoteOn = 0x90,
    PolyPressure = 0xA0,
    ControlChange = 0xB0,
    ProgramChange = 0xC0,
    ChannelPressure = 0xD0,
    PitchBend = 0xE0,
    System = 0xF0
};

/// Compact MIDI event (for real-time use)
struct alignas(8) MidiEvent {
    Tick timestamp = 0;
    uint8_t status = 0;
    uint8_t data1 = 0;
    uint8_t data2 = 0;
    uint8_t padding = 0;

    MidiEvent() = default;

    MidiEvent(Tick ts, MidiMessageType type, MidiChannel ch, uint8_t d1, uint8_t d2)
        : timestamp(ts)
        , status(static_cast<uint8_t>(type) | (ch & 0x0F))
        , data1(d1)
        , data2(d2)
        , padding(0)
    {}

    MidiMessageType type() const {
        return static_cast<MidiMessageType>(status & 0xF0);
    }

    MidiChannel channel() const {
        return status & 0x0F;
    }

    bool isNoteOn() const {
        return type() == MidiMessageType::NoteOn && data2 > 0;
    }

    bool isNoteOff() const {
        return type() == MidiMessageType::NoteOff ||
               (type() == MidiMessageType::NoteOn && data2 == 0);
    }
};

// =============================================================================
// Utility Structures
// =============================================================================

/**
 * @brief Time signature representation
 */
struct TimeSignature {
    uint8_t numerator = 4;
    uint8_t denominator = 4;

    [[nodiscard]] constexpr float beatsPerBar() const {
        return static_cast<float>(numerator);
    }

    [[nodiscard]] constexpr float ticksPerBar(int ppq) const {
        return beatsPerBar() * static_cast<float>(ppq) *
               (4.0f / static_cast<float>(denominator));
    }
};

/**
 * @brief Tempo in BPM with micro-timing support
 */
struct Tempo {
    float bpm = 120.0f;

    [[nodiscard]] constexpr float samplesPerBeat(SampleRate sr) const {
        return (60.0f / bpm) * static_cast<float>(sr);
    }

    [[nodiscard]] constexpr float msPerBeat() const {
        return 60000.0f / bpm;
    }
};

/**
 * @brief Note event for MIDI/audio rendering
 */
struct NoteEvent {
    MidiNote pitch = 60;
    MidiVelocity velocity = 100;
    TickCount startTick = 0;
    TickCount durationTicks = 480;
    MidiChannel channel = 0;

    [[nodiscard]] constexpr TickCount endTick() const {
        return startTick + durationTicks;
    }
};

/**
 * @brief Groove template for timing adjustments
 */
struct GrooveSettings {
    float swing = 0.0f;         // 0.0 = straight, 0.5 = full swing
    float pushPull = 0.0f;      // -1.0 = behind, +1.0 = ahead
    float humanization = 0.0f;  // Random timing variance (0.0-1.0)
    float velocityVar = 0.0f;   // Random velocity variance (0.0-1.0)
};

// =============================================================================
// Version Info
// =============================================================================

#ifndef DAIW_VERSION_MAJOR
#define DAIW_VERSION_MAJOR 1
#endif

#ifndef DAIW_VERSION_MINOR
#define DAIW_VERSION_MINOR 0
#endif

#ifndef DAIW_VERSION_PATCH
#define DAIW_VERSION_PATCH 0
#endif

struct Version {
    static constexpr int major = DAIW_VERSION_MAJOR;
    static constexpr int minor = DAIW_VERSION_MINOR;
    static constexpr int patch = DAIW_VERSION_PATCH;

    // Version string macro for compile-time construction
    #define DAIW_STRINGIFY(x) #x
    #define DAIW_VERSION_TO_STRING(major, minor, patch) \
        DAIW_STRINGIFY(major) "." DAIW_STRINGIFY(minor) "." DAIW_STRINGIFY(patch)

    static constexpr const char* string() {
        return DAIW_VERSION_TO_STRING(DAIW_VERSION_MAJOR, DAIW_VERSION_MINOR, DAIW_VERSION_PATCH);
    }
};

}  // namespace daiw
