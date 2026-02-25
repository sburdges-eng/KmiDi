/**
 * @file MidiSequence.h
 * @brief MIDI sequence container with timing and manipulation
 *
 * Provides a time-ordered container for MIDI messages with:
 * - Automatic sorting by timestamp
 * - Quantization support
 * - Event filtering and manipulation
 * - Conversion to/from NoteEvent structures
 */

#pragma once

#include "daiw/midi/MidiMessage.h"
#include "daiw/types.hpp"
#include <vector>
#include <algorithm>

namespace daiw {
namespace midi {

/**
 * @brief Container for a time-ordered sequence of MIDI messages
 */
class MidiSequence {
public:
    /**
     * @brief Default constructor
     */
    MidiSequence(int ppq = DEFAULT_PPQ) : ppq_(ppq) {}

    /**
     * @brief Add a MIDI message to the sequence
     * @param message The MIDI message to add
     * @param autoSort If true, maintain sorted order (default: false for batch adds)
     */
    void addMessage(const MidiMessage& message, bool autoSort = false) {
        messages_.push_back(message);
        if (autoSort) {
            sort();
        }
    }

    /**
     * @brief Add multiple messages at once
     */
    void addMessages(const std::vector<MidiMessage>& messages) {
        messages_.reserve(messages_.size() + messages.size());
        messages_.insert(messages_.end(), messages.begin(), messages.end());
    }

    /**
     * @brief Sort messages by timestamp
     */
    void sort() {
        std::sort(messages_.begin(), messages_.end(),
            [](const MidiMessage& a, const MidiMessage& b) {
                return a.getTimestamp() < b.getTimestamp();
            });
    }

    /**
     * @brief Clear all messages
     */
    void clear() {
        messages_.clear();
    }

    /**
     * @brief Get all messages
     */
    [[nodiscard]] const std::vector<MidiMessage>& getMessages() const {
        return messages_;
    }

    /**
     * @brief Get number of messages
     */
    [[nodiscard]] size_t size() const {
        return messages_.size();
    }

    /**
     * @brief Check if sequence is empty
     */
    [[nodiscard]] bool empty() const {
        return messages_.empty();
    }

    /**
     * @brief Get PPQ (Pulses Per Quarter note)
     */
    [[nodiscard]] int getPPQ() const {
        return ppq_;
    }

    /**
     * @brief Set PPQ
     */
    void setPPQ(int ppq) {
        ppq_ = ppq;
    }

    /**
     * @brief Quantize all note events to the grid
     * @param gridSize Grid size in ticks (e.g., ppq/4 for 16th notes)
     */
    void quantize(TickCount gridSize) {
        if (gridSize == 0) {
            return;
        }
        for (auto& msg : messages_) {
            if (msg.isNoteOn() || msg.isNoteOff()) {
                TickCount timestamp = msg.getTimestamp();
                TickCount quantized = ((timestamp + gridSize / 2) / gridSize) * gridSize;
                msg.setTimestamp(quantized);
            }
        }
        sort();
    }

    /**
     * @brief Get all note on messages
     */
    [[nodiscard]] std::vector<MidiMessage> getNoteOnMessages() const {
        std::vector<MidiMessage> noteOns;
        for (const auto& msg : messages_) {
            if (msg.isNoteOn()) {
                noteOns.push_back(msg);
            }
        }
        return noteOns;
    }

    /**
     * @brief Get all note off messages
     */
    [[nodiscard]] std::vector<MidiMessage> getNoteOffMessages() const {
        std::vector<MidiMessage> noteOffs;
        for (const auto& msg : messages_) {
            if (msg.isNoteOff()) {
                noteOffs.push_back(msg);
            }
        }
        return noteOffs;
    }

    /**
     * @brief Convert to NoteEvent structures (pairs note on/off)
     */
    [[nodiscard]] std::vector<NoteEvent> toNoteEvents() const;

    /**
     * @brief Create sequence from NoteEvent structures
     */
    static MidiSequence fromNoteEvents(const std::vector<NoteEvent>& events, int ppq = DEFAULT_PPQ);

    /**
     * @brief Get messages in a time range
     * @param startTick Start time (inclusive)
     * @param endTick End time (exclusive)
     */
    [[nodiscard]] std::vector<MidiMessage> getMessagesInRange(
        TickCount startTick, TickCount endTick) const {
        std::vector<MidiMessage> result;
        for (const auto& msg : messages_) {
            TickCount t = msg.getTimestamp();
            if (t >= startTick && t < endTick) {
                result.push_back(msg);
            }
        }
        return result;
    }

    /**
     * @brief Filter messages by type
     */
    [[nodiscard]] std::vector<MidiMessage> filterByType(MessageType type) const {
        std::vector<MidiMessage> result;
        for (const auto& msg : messages_) {
            if (msg.getType() == type) {
                result.push_back(msg);
            }
        }
        return result;
    }

    /**
     * @brief Filter messages by channel
     */
    [[nodiscard]] std::vector<MidiMessage> filterByChannel(MidiChannel channel) const {
        std::vector<MidiMessage> result;
        for (const auto& msg : messages_) {
            if (msg.getChannel() == channel) {
                result.push_back(msg);
            }
        }
        return result;
    }

    /**
     * @brief Transpose all note events by semitones
     */
    void transpose(int semitones) {
        for (auto& msg : messages_) {
            if (msg.isNoteOn() || msg.isNoteOff()) {
                int newNote = static_cast<int>(msg.getNoteNumber()) + semitones;
                if (newNote >= MIDI_NOTE_MIN && newNote <= MIDI_NOTE_MAX) {
                    // Create new message with transposed note
                    MidiMessage newMsg = msg;
                    if (msg.isNoteOn()) {
                        newMsg = MidiMessage::noteOn(
                            msg.getChannel(),
                            static_cast<MidiNote>(newNote),
                            msg.getVelocity()
                        );
                    } else {
                        newMsg = MidiMessage::noteOff(
                            msg.getChannel(),
                            static_cast<MidiNote>(newNote),
                            msg.getVelocity()
                        );
                    }
                    newMsg.setTimestamp(msg.getTimestamp());
                    msg = newMsg;
                }
            }
        }
    }

    /**
     * @brief Get duration of sequence in ticks
     */
    [[nodiscard]] TickCount getDuration() const {
        if (messages_.empty()) {
            return 0;
        }
        return messages_.back().getTimestamp();
    }

private:
    std::vector<MidiMessage> messages_;
    int ppq_;
};

} // namespace midi
} // namespace daiw
