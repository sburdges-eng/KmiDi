#pragma once

// daiw/rt_event.hpp — POD event type for the realtime scheduler.
//
// Single allocation-free record covering MIDI mutation (note/CC/AT/PB/PC),
// transport ops (play/stop/seek/tempo), and panic clears (AllNotesOff,
// AllSoundOff). Designed for lock-free MPSC queues and a min-heap drain on
// the audio thread.

#include <cstdint>

namespace daiw::rt {

enum class EventKind : uint8_t {
  // MIDI channel-voice messages
  NoteOn = 1,
  NoteOff = 2,
  PolyPressure = 3,
  ControlChange = 4,
  ProgramChange = 5,
  ChannelPressure = 6,
  PitchBend = 7,
  // Transport / clock
  TransportPlay = 8,
  TransportStop = 9,
  TransportSeek = 10,
  TempoChange = 11,
  // Panic / mutation helpers
  AllNotesOff = 12,
  AllSoundOff = 13,
};

// 16-byte POD. trivially copyable + standard-layout so it survives any
// lock-free queue without UB.
struct alignas(16) Event {
  uint64_t sample_time = 0; // absolute sample position (host clock)
  EventKind kind = EventKind::NoteOff;
  uint8_t channel = 0; // 0..15 (MIDI), 0 for transport/tempo
  uint8_t data1 = 0;   // note#, CC#, program#
  uint8_t data2 = 0;   // velocity, CC value, pressure
  // Wide payload — pitch-bend (14-bit), tempo (bpm * 100), seek (low 16 of
  // target sample offset). For larger payloads, use multiple events.
  uint16_t data16 = 0;
  uint16_t _reserved = 0;
};

static_assert(sizeof(Event) == 16, "Event must remain a 16-byte POD");

// Builder helpers (free functions; keep Event itself an aggregate POD).
constexpr Event make_note_on(uint64_t t, uint8_t ch, uint8_t note,
                             uint8_t vel) noexcept {
  return Event{t, EventKind::NoteOn, ch, note, vel, 0, 0};
}
constexpr Event make_note_off(uint64_t t, uint8_t ch, uint8_t note,
                              uint8_t vel = 0) noexcept {
  return Event{t, EventKind::NoteOff, ch, note, vel, 0, 0};
}
constexpr Event make_cc(uint64_t t, uint8_t ch, uint8_t cc,
                        uint8_t value) noexcept {
  return Event{t, EventKind::ControlChange, ch, cc, value, 0, 0};
}
constexpr Event make_aftertouch(uint64_t t, uint8_t ch,
                                uint8_t pressure) noexcept {
  return Event{t, EventKind::ChannelPressure, ch, pressure, 0, 0, 0};
}
constexpr Event make_pitch_bend(uint64_t t, uint8_t ch,
                                uint16_t bend14) noexcept {
  return Event{t,
               EventKind::PitchBend,
               ch,
               static_cast<uint8_t>(bend14 & 0x7F),
               static_cast<uint8_t>((bend14 >> 7) & 0x7F),
               bend14,
               0};
}
constexpr Event make_tempo_change(uint64_t t, float bpm) noexcept {
  return Event{
      t, EventKind::TempoChange, 0, 0, 0, static_cast<uint16_t>(bpm * 100.0f),
      0};
}
constexpr Event make_all_notes_off(uint64_t t, uint8_t ch) noexcept {
  return Event{t, EventKind::AllNotesOff, ch, 0, 0, 0, 0};
}

} // namespace daiw::rt
