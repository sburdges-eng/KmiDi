/**
 * @file MidiMessage.h
 * @brief MIDI message types and utilities
 *
 * Provides comprehensive MIDI message representation including:
 * - Note On/Off
 * - Control Change (CC)
 * - Pitch Bend
 * - Program Change
 * - System messages
 */

#pragma once

#include "daiw/types.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace daiw {
namespace midi {

/**
 * @brief MIDI message type enumeration
 */
enum class MessageType : uint8_t {
    NoteOff = 0x80,
    NoteOn = 0x90,
    PolyPressure = 0xA0,
    ControlChange = 0xB0,
    ProgramChange = 0xC0,
    ChannelPressure = 0xD0,
    PitchBend = 0xE0,
    SystemExclusive = 0xF0,
    TimeCode = 0xF1,
    SongPosition = 0xF2,
    SongSelect = 0xF3,
    TuneRequest = 0xF6,
    Clock = 0xF8,
    Start = 0xFA,
    Continue = 0xFB,
    Stop = 0xFC,
    ActiveSensing = 0xFE,
    SystemReset = 0xFF,
    Invalid = 0x00
};

/**
 * @brief Common MIDI Control Change numbers
 */
namespace CC {
    constexpr uint8_t BankSelect = 0;
    constexpr uint8_t ModWheel = 1;
    constexpr uint8_t Breath = 2;
    constexpr uint8_t FootController = 4;
    constexpr uint8_t PortamentoTime = 5;
    constexpr uint8_t Volume = 7;
    constexpr uint8_t Balance = 8;
    constexpr uint8_t Pan = 10;
    constexpr uint8_t Expression = 11;
    constexpr uint8_t Sustain = 64;
    constexpr uint8_t Portamento = 65;
    constexpr uint8_t Sostenuto = 66;
    constexpr uint8_t SoftPedal = 67;
    constexpr uint8_t Legato = 68;
    constexpr uint8_t AllSoundOff = 120;
    constexpr uint8_t ResetAllControllers = 121;
    constexpr uint8_t AllNotesOff = 123;
}

/**
 * @brief MIDI message representation
 *
 * Efficient representation of MIDI messages with timestamp support.
 * Supports all standard MIDI message types.
 */
class MidiMessage {
public:
    /**
     * @brief Default constructor - creates invalid message
     */
    MidiMessage() : timestamp_(0), data_{0, 0, 0} {}

    /**
     * @brief Create a Note On message
     */
    static MidiMessage noteOn(MidiChannel channel, MidiNote note, MidiVelocity velocity) {
        MidiMessage msg;
        msg.data_[0] = static_cast<uint8_t>(MessageType::NoteOn) | (channel & 0x0F);
        msg.data_[1] = note & 0x7F;
        msg.data_[2] = velocity & 0x7F;
        return msg;
    }

    /**
     * @brief Create a Note Off message
     */
    static MidiMessage noteOff(MidiChannel channel, MidiNote note, MidiVelocity velocity = 0) {
        MidiMessage msg;
        msg.data_[0] = static_cast<uint8_t>(MessageType::NoteOff) | (channel & 0x0F);
        msg.data_[1] = note & 0x7F;
        msg.data_[2] = velocity & 0x7F;
        return msg;
    }

    /**
     * @brief Create a Control Change message
     */
    static MidiMessage controlChange(MidiChannel channel, uint8_t controller, uint8_t value) {
        MidiMessage msg;
        msg.data_[0] = static_cast<uint8_t>(MessageType::ControlChange) | (channel & 0x0F);
        msg.data_[1] = controller & 0x7F;
        msg.data_[2] = value & 0x7F;
        return msg;
    }

    /**
     * @brief Create a Pitch Bend message
     * @param channel MIDI channel (0-15)
     * @param value Pitch bend value (0-16383, center is 8192)
     */
    static MidiMessage pitchBend(MidiChannel channel, uint16_t value) {
        MidiMessage msg;
        msg.data_[0] = static_cast<uint8_t>(MessageType::PitchBend) | (channel & 0x0F);
        msg.data_[1] = value & 0x7F;         // LSB
        msg.data_[2] = (value >> 7) & 0x7F;  // MSB
        return msg;
    }

    /**
     * @brief Create a Program Change message
     */
    static MidiMessage programChange(MidiChannel channel, uint8_t program) {
        MidiMessage msg;
        msg.data_[0] = static_cast<uint8_t>(MessageType::ProgramChange) | (channel & 0x0F);
        msg.data_[1] = program & 0x7F;
        msg.data_[2] = 0;
        return msg;
    }

    /**
     * @brief Create a Channel Pressure (aftertouch) message
     */
    static MidiMessage channelPressure(MidiChannel channel, uint8_t pressure) {
        MidiMessage msg;
        msg.data_[0] = static_cast<uint8_t>(MessageType::ChannelPressure) | (channel & 0x0F);
        msg.data_[1] = pressure & 0x7F;
        msg.data_[2] = 0;
        return msg;
    }

    // Getters
    [[nodiscard]] MessageType getType() const {
        return static_cast<MessageType>(data_[0] & 0xF0);
    }

    [[nodiscard]] MidiChannel getChannel() const {
        return data_[0] & 0x0F;
    }

    [[nodiscard]] uint8_t getStatusByte() const {
        return data_[0];
    }

    [[nodiscard]] uint8_t getData1() const {
        return data_[1];
    }

    [[nodiscard]] uint8_t getData2() const {
        return data_[2];
    }

    [[nodiscard]] TickCount getTimestamp() const {
        return timestamp_;
    }

    void setTimestamp(TickCount timestamp) {
        timestamp_ = timestamp;
    }

    // Type checking
    [[nodiscard]] bool isNoteOn() const {
        return getType() == MessageType::NoteOn && data_[2] > 0;
    }

    [[nodiscard]] bool isNoteOff() const {
        return getType() == MessageType::NoteOff || 
               (getType() == MessageType::NoteOn && data_[2] == 0);
    }

    [[nodiscard]] bool isControlChange() const {
        return getType() == MessageType::ControlChange;
    }

    [[nodiscard]] bool isPitchBend() const {
        return getType() == MessageType::PitchBend;
    }

    [[nodiscard]] bool isProgramChange() const {
        return getType() == MessageType::ProgramChange;
    }

    // Note-specific getters
    [[nodiscard]] MidiNote getNoteNumber() const {
        return data_[1];
    }

    [[nodiscard]] MidiVelocity getVelocity() const {
        return data_[2];
    }

    // Control Change specific
    [[nodiscard]] uint8_t getControllerNumber() const {
        return data_[1];
    }

    [[nodiscard]] uint8_t getControllerValue() const {
        return data_[2];
    }

    // Pitch Bend specific (returns 0-16383, center is 8192)
    [[nodiscard]] uint16_t getPitchBendValue() const {
        return (static_cast<uint16_t>(data_[2]) << 7) | data_[1];
    }

    // Utility
    [[nodiscard]] std::string toString() const;

private:
    TickCount timestamp_;   // Timestamp in ticks
    uint8_t data_[3];       // Status byte + up to 2 data bytes
};

} // namespace midi
} // namespace daiw
