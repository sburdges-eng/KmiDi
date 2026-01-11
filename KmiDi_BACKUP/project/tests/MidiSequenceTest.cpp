/**
 * @file MidiSequenceTest.cpp
 * @brief Unit tests for MIDI message and sequence classes
 */

#include <gtest/gtest.h>
#include "daiw/midi/MidiMessage.h"
#include "daiw/midi/MidiSequence.h"

using namespace daiw;
using namespace daiw::midi;

// ============================================================================
// MidiMessage Tests
// ============================================================================

TEST(MidiMessageTest, CreateNoteOn) {
    auto msg = MidiMessage::noteOn(0, 60, 100);
    
    EXPECT_TRUE(msg.isNoteOn());
    EXPECT_FALSE(msg.isNoteOff());
    EXPECT_EQ(msg.getChannel(), 0);
    EXPECT_EQ(msg.getNoteNumber(), 60);
    EXPECT_EQ(msg.getVelocity(), 100);
}

TEST(MidiMessageTest, CreateNoteOff) {
    auto msg = MidiMessage::noteOff(1, 64, 0);
    
    EXPECT_TRUE(msg.isNoteOff());
    EXPECT_FALSE(msg.isNoteOn());
    EXPECT_EQ(msg.getChannel(), 1);
    EXPECT_EQ(msg.getNoteNumber(), 64);
}

TEST(MidiMessageTest, NoteOnZeroVelocityIsNoteOff) {
    auto msg = MidiMessage::noteOn(0, 60, 0);
    
    EXPECT_FALSE(msg.isNoteOn());
    EXPECT_TRUE(msg.isNoteOff());
}

TEST(MidiMessageTest, ControlChange) {
    auto msg = MidiMessage::controlChange(2, CC::ModWheel, 64);
    
    EXPECT_TRUE(msg.isControlChange());
    EXPECT_EQ(msg.getChannel(), 2);
    EXPECT_EQ(msg.getControllerNumber(), CC::ModWheel);
    EXPECT_EQ(msg.getControllerValue(), 64);
}

TEST(MidiMessageTest, PitchBend) {
    auto msg = MidiMessage::pitchBend(0, 8192);  // Center position
    
    EXPECT_TRUE(msg.isPitchBend());
    EXPECT_EQ(msg.getPitchBendValue(), 8192);
    EXPECT_EQ(msg.getChannel(), 0);
}

TEST(MidiMessageTest, PitchBendMinMax) {
    auto minMsg = MidiMessage::pitchBend(0, 0);
    EXPECT_EQ(minMsg.getPitchBendValue(), 0);
    
    auto maxMsg = MidiMessage::pitchBend(0, 16383);
    EXPECT_EQ(maxMsg.getPitchBendValue(), 16383);
}

TEST(MidiMessageTest, ProgramChange) {
    auto msg = MidiMessage::programChange(3, 42);
    
    EXPECT_TRUE(msg.isProgramChange());
    EXPECT_EQ(msg.getChannel(), 3);
    EXPECT_EQ(msg.getData1(), 42);
}

TEST(MidiMessageTest, ChannelPressure) {
    auto msg = MidiMessage::channelPressure(0, 80);
    
    EXPECT_EQ(msg.getChannel(), 0);
    EXPECT_EQ(msg.getData1(), 80);
}

TEST(MidiMessageTest, Timestamp) {
    auto msg = MidiMessage::noteOn(0, 60, 100);
    
    EXPECT_EQ(msg.getTimestamp(), 0);
    
    msg.setTimestamp(480);
    EXPECT_EQ(msg.getTimestamp(), 480);
}

TEST(MidiMessageTest, ToString) {
    auto msg = MidiMessage::noteOn(0, 60, 100);
    msg.setTimestamp(0);
    
    std::string str = msg.toString();
    EXPECT_FALSE(str.empty());
    EXPECT_NE(str.find("Note On"), std::string::npos);
}

// ============================================================================
// MidiSequence Tests
// ============================================================================

TEST(MidiSequenceTest, CreateEmpty) {
    MidiSequence seq;
    
    EXPECT_TRUE(seq.empty());
    EXPECT_EQ(seq.size(), 0);
    EXPECT_EQ(seq.getPPQ(), DEFAULT_PPQ);
}

TEST(MidiSequenceTest, AddMessages) {
    MidiSequence seq;
    
    auto msg1 = MidiMessage::noteOn(0, 60, 100);
    msg1.setTimestamp(0);
    seq.addMessage(msg1);
    
    auto msg2 = MidiMessage::noteOff(0, 60, 0);
    msg2.setTimestamp(480);
    seq.addMessage(msg2);
    
    EXPECT_EQ(seq.size(), 2);
    EXPECT_FALSE(seq.empty());
}

TEST(MidiSequenceTest, Sorting) {
    MidiSequence seq;
    
    // Add messages out of order
    auto msg1 = MidiMessage::noteOn(0, 60, 100);
    msg1.setTimestamp(480);
    seq.addMessage(msg1);
    
    auto msg2 = MidiMessage::noteOn(0, 64, 100);
    msg2.setTimestamp(0);
    seq.addMessage(msg2);
    
    auto msg3 = MidiMessage::noteOn(0, 67, 100);
    msg3.setTimestamp(240);
    seq.addMessage(msg3);
    
    seq.sort();
    
    const auto& messages = seq.getMessages();
    EXPECT_EQ(messages[0].getTimestamp(), 0);
    EXPECT_EQ(messages[1].getTimestamp(), 240);
    EXPECT_EQ(messages[2].getTimestamp(), 480);
}

TEST(MidiSequenceTest, Quantization) {
    MidiSequence seq(480);
    
    // Add slightly off-grid notes
    auto msg1 = MidiMessage::noteOn(0, 60, 100);
    msg1.setTimestamp(125);  // Slightly after beat 1
    seq.addMessage(msg1);
    
    auto msg2 = MidiMessage::noteOn(0, 64, 100);
    msg2.setTimestamp(360);  // Slightly before beat 2
    seq.addMessage(msg2);
    
    // Quantize to 16th notes (PPQ / 4)
    seq.quantize(480 / 4);
    
    const auto& messages = seq.getMessages();
    EXPECT_EQ(messages[0].getTimestamp(), 120);  // Quantized to 16th
    EXPECT_EQ(messages[1].getTimestamp(), 360);  // Quantized to 16th
}

TEST(MidiSequenceTest, GetNoteOnMessages) {
    MidiSequence seq;
    
    seq.addMessage(MidiMessage::noteOn(0, 60, 100));
    seq.addMessage(MidiMessage::noteOff(0, 60, 0));
    seq.addMessage(MidiMessage::noteOn(0, 64, 100));
    seq.addMessage(MidiMessage::controlChange(0, CC::Volume, 100));
    
    auto noteOns = seq.getNoteOnMessages();
    EXPECT_EQ(noteOns.size(), 2);
}

TEST(MidiSequenceTest, GetNoteOffMessages) {
    MidiSequence seq;
    
    seq.addMessage(MidiMessage::noteOn(0, 60, 100));
    seq.addMessage(MidiMessage::noteOff(0, 60, 0));
    seq.addMessage(MidiMessage::noteOn(0, 64, 0));  // Zero velocity = note off
    
    auto noteOffs = seq.getNoteOffMessages();
    EXPECT_EQ(noteOffs.size(), 2);
}

TEST(MidiSequenceTest, ToNoteEvents) {
    MidiSequence seq(480);
    
    // Add a complete note (on + off)
    auto noteOn = MidiMessage::noteOn(0, 60, 100);
    noteOn.setTimestamp(0);
    seq.addMessage(noteOn);
    
    auto noteOff = MidiMessage::noteOff(0, 60, 0);
    noteOff.setTimestamp(480);
    seq.addMessage(noteOff);
    
    auto events = seq.toNoteEvents();
    
    ASSERT_EQ(events.size(), 1);
    EXPECT_EQ(events[0].pitch, 60);
    EXPECT_EQ(events[0].velocity, 100);
    EXPECT_EQ(events[0].startTick, 0);
    EXPECT_EQ(events[0].durationTicks, 480);
    EXPECT_EQ(events[0].channel, 0);
}

TEST(MidiSequenceTest, FromNoteEvents) {
    std::vector<NoteEvent> events;
    
    NoteEvent event1;
    event1.pitch = 60;
    event1.velocity = 100;
    event1.startTick = 0;
    event1.durationTicks = 480;
    event1.channel = 0;
    events.push_back(event1);
    
    NoteEvent event2;
    event2.pitch = 64;
    event2.velocity = 90;
    event2.startTick = 480;
    event2.durationTicks = 480;
    event2.channel = 0;
    events.push_back(event2);
    
    auto seq = MidiSequence::fromNoteEvents(events, 480);
    
    EXPECT_EQ(seq.size(), 4);  // 2 note ons + 2 note offs
    EXPECT_EQ(seq.getNoteOnMessages().size(), 2);
    EXPECT_EQ(seq.getNoteOffMessages().size(), 2);
}

TEST(MidiSequenceTest, GetMessagesInRange) {
    MidiSequence seq;
    
    auto msg1 = MidiMessage::noteOn(0, 60, 100);
    msg1.setTimestamp(0);
    seq.addMessage(msg1);
    
    auto msg2 = MidiMessage::noteOn(0, 64, 100);
    msg2.setTimestamp(480);
    seq.addMessage(msg2);
    
    auto msg3 = MidiMessage::noteOn(0, 67, 100);
    msg3.setTimestamp(960);
    seq.addMessage(msg3);
    
    auto range = seq.getMessagesInRange(400, 800);
    ASSERT_EQ(range.size(), 1);
    EXPECT_EQ(range[0].getTimestamp(), 480);
}

TEST(MidiSequenceTest, FilterByType) {
    MidiSequence seq;
    
    seq.addMessage(MidiMessage::noteOn(0, 60, 100));
    seq.addMessage(MidiMessage::controlChange(0, CC::Volume, 100));
    seq.addMessage(MidiMessage::noteOn(0, 64, 100));
    seq.addMessage(MidiMessage::pitchBend(0, 8192));
    
    auto noteOns = seq.filterByType(MessageType::NoteOn);
    EXPECT_EQ(noteOns.size(), 2);
    
    auto ccs = seq.filterByType(MessageType::ControlChange);
    EXPECT_EQ(ccs.size(), 1);
}

TEST(MidiSequenceTest, FilterByChannel) {
    MidiSequence seq;
    
    seq.addMessage(MidiMessage::noteOn(0, 60, 100));
    seq.addMessage(MidiMessage::noteOn(1, 64, 100));
    seq.addMessage(MidiMessage::noteOn(0, 67, 100));
    seq.addMessage(MidiMessage::noteOn(2, 72, 100));
    
    auto ch0 = seq.filterByChannel(0);
    EXPECT_EQ(ch0.size(), 2);
    
    auto ch1 = seq.filterByChannel(1);
    EXPECT_EQ(ch1.size(), 1);
}

TEST(MidiSequenceTest, Transpose) {
    MidiSequence seq;
    
    auto msg1 = MidiMessage::noteOn(0, 60, 100);
    seq.addMessage(msg1);
    
    auto msg2 = MidiMessage::noteOn(0, 64, 100);
    seq.addMessage(msg2);
    
    seq.transpose(12);  // Up one octave
    
    const auto& messages = seq.getMessages();
    EXPECT_EQ(messages[0].getNoteNumber(), 72);
    EXPECT_EQ(messages[1].getNoteNumber(), 76);
}

TEST(MidiSequenceTest, TransposeClipping) {
    MidiSequence seq;
    
    auto msg = MidiMessage::noteOn(0, 120, 100);
    seq.addMessage(msg);
    
    seq.transpose(10);  // Would exceed 127
    
    // Note should remain at 120 (not transposed beyond limit)
    const auto& messages = seq.getMessages();
    EXPECT_EQ(messages[0].getNoteNumber(), 120);
}

TEST(MidiSequenceTest, GetDuration) {
    MidiSequence seq;
    
    EXPECT_EQ(seq.getDuration(), 0);
    
    auto msg1 = MidiMessage::noteOn(0, 60, 100);
    msg1.setTimestamp(0);
    seq.addMessage(msg1);
    
    auto msg2 = MidiMessage::noteOff(0, 60, 0);
    msg2.setTimestamp(1920);
    seq.addMessage(msg2);
    
    seq.sort();
    
    EXPECT_EQ(seq.getDuration(), 1920);
}

// Run all tests
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
