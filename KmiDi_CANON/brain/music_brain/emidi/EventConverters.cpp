#include "EventConverters.hpp"

namespace emidi {

Event from_note_event(
    const music_brain::NoteEvent& note,
    const std::string& intent_id,
    const std::string& rule_break,
    const std::string& emotional_effect
) {
    Event event;
    event.note = static_cast<std::uint8_t>(note.pitch);
    event.velocity = static_cast<std::uint8_t>(note.velocity);
    event.start_tick = static_cast<std::uint32_t>(note.start_tick);
    event.duration_ticks = static_cast<std::uint32_t>(note.duration_ticks);
    event.intent_id = intent_id;
    event.rule_break = rule_break;
    event.emotional_effect = emotional_effect;
    return event;
}

std::vector<music_brain::NoteEvent> to_note_events(const std::vector<Event>& events) {
    std::vector<music_brain::NoteEvent> notes;
    notes.reserve(events.size());
    for (const auto& event : events) {
        notes.push_back(music_brain::NoteEvent{
            static_cast<int>(event.note),
            static_cast<int>(event.velocity),
            static_cast<int>(event.start_tick),
            static_cast<int>(event.duration_ticks),
        });
    }
    return notes;
}

}  // namespace emidi

