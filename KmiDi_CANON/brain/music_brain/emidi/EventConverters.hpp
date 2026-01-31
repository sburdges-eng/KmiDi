#pragma once

#include "Event.hpp"
#include "NoteEventStub.hpp"

#include <vector>

namespace emidi {

/**
 * Convert a NoteEvent from the existing C++/Python bridge into an EMIDI Event.
 * Emotional metadata can be injected via parameters or defaults.
 */
Event from_note_event(
    const music_brain::NoteEvent& note,
    const std::string& intent_id = "",
    const std::string& rule_break = "",
    const std::string& emotional_effect = ""
);

/**
 * Convert EMIDI events back to NoteEvents for legacy code paths.
 */
std::vector<music_brain::NoteEvent> to_note_events(const std::vector<Event>& events);

}  // namespace emidi

