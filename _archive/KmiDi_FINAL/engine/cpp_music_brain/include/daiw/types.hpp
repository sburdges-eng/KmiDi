#pragma once

#include <cstdint>
#include <cstddef>
#include <array>
#include <span>
#include <string>

namespace daiw {

// =============================================================================
// Basic Types
// =============================================================================

using SampleRate = uint32_t;
using BlockSize = uint32_t;
using ChannelCount = uint8_t;
using MidiChannel = uint8_t;
using MidiNote = uint8_t;
using MidiVelocity = uint8_t;
using Tick = int64_t;  // MIDI ticks (PPQ-based)
using TickCount = Tick;

/// Pulses per quarter note (PPQ) default
constexpr int DEFAULT_PPQ = 480;

/// Tempo container
struct Tempo {
    float bpm {120.0f};

    double samplesPerBeat(double sampleRate) const noexcept {
        return (sampleRate * 60.0) / static_cast<double>(bpm > 0 ? bpm : 120.0f);
    }

    double msPerBeat() const noexcept {
        return 60000.0 / static_cast<double>(bpm > 0 ? bpm : 120.0f);
    }
};

struct TimeSignature {
    int numerator {4};
    int denominator {4};

    int beatsPerBar() const noexcept { return numerator; }
    Tick ticksPerBar(int ppq = DEFAULT_PPQ) const noexcept {
        return static_cast<Tick>(ppq) * numerator * 4 / (denominator > 0 ? denominator : 4);
    }
};

struct NoteEvent {
    MidiNote pitch {0};
    MidiVelocity velocity {0};
    Tick startTick {0};
    Tick durationTicks {0};
    MidiChannel channel {0};

    Tick endTick() const noexcept { return startTick + durationTicks; }
};

struct GrooveSettings {
    float swing {0.0f};
    float pushPull {0.0f};
    float humanization {0.0f};
    float velocityVar {0.0f};
};

struct Version {
    static const char* string() noexcept { return "0.0.0"; }
};

// =============================================================================
// Audio Types
// =============================================================================

/// Single audio sample (32-bit float)
using Sample = float;

/// Stereo sample pair
struct StereoSample {
    Sample left;
    Sample right;

    StereoSample() : left(0.0f), right(0.0f) {}
    StereoSample(Sample l, Sample r) : left(l), right(r) {}
    StereoSample(Sample mono) : left(mono), right(mono) {}

    StereoSample operator+(const StereoSample& other) const {
        return {left + other.left, right + other.right};
    }

    StereoSample operator*(Sample gain) const {
        return {left * gain, right * gain};
    }
};

/// Audio buffer view (non-owning)
using AudioSpan = std::span<Sample>;
using ConstAudioSpan = std::span<const Sample>;

// =============================================================================
// MIDI Types
// =============================================================================

/// MIDI message types
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

/// Compact MIDI event (8 bytes)
struct alignas(8) MidiEvent {
    Tick timestamp;          // 8 bytes - when to play
    uint8_t status;          // 1 byte - message type + channel
    uint8_t data1;           // 1 byte - note/CC number
    uint8_t data2;           // 1 byte - velocity/value
    uint8_t padding;         // 1 byte - alignment

    MidiEvent() : timestamp(0), status(0), data1(0), data2(0), padding(0) {}

    MidiEvent(Tick ts, MidiMessageType type, MidiChannel channel, uint8_t d1, uint8_t d2)
        : timestamp(ts)
        , status(static_cast<uint8_t>(type) | (channel & 0x0F))
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

static_assert(sizeof(MidiEvent) == 16, "MidiEvent must be 16 bytes");

// =============================================================================
// Groove Types
// =============================================================================

/// Timing offset in ticks (can be negative for "behind the beat")
using TimingOffset = int16_t;

/// Groove template entry
struct GroovePoint {
    TimingOffset timing;     // Offset from grid in ticks
    uint8_t velocity_scale;  // Velocity multiplier (0-200, 100 = no change)
    uint8_t probability;     // Probability of playing (0-100)

    GroovePoint() : timing(0), velocity_scale(100), probability(100) {}
    GroovePoint(TimingOffset t, uint8_t v = 100, uint8_t p = 100)
        : timing(t), velocity_scale(v), probability(p) {}
};

/// Fixed-size groove template (16 steps)
using GrooveTemplate = std::array<GroovePoint, 16>;

// =============================================================================
// Processing Context
// =============================================================================

/// Audio processing context passed to processors
struct ProcessContext {
    SampleRate sample_rate;
    BlockSize block_size;
    double bpm;
    double beat_position;    // Current position in beats
    bool is_playing;
    bool transport_changed;  // True if transport state changed this block
};

} // namespace daiw
