#pragma once

namespace music_brain {

/**
 * Temporary NoteEvent definition for the C++ migration.
 * Replace with the real structure once the C++ NoteEvent becomes available.
 */
struct NoteEvent {
    int pitch = 60;
    int velocity = 80;
    int start_tick = 0;
    int duration_ticks = 480;
};

}  // namespace music_brain

