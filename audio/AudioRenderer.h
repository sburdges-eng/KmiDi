#pragma once

/**
 * D2 — MIDI → PCM renderer.
 * Consumes GeneratedMidi only. No intent, no generation state.
 * Used exclusively for visualization (read-only waveform display).
 */

#include "common/Types.h"
#include <vector>

namespace kelly {

/** Render validated output plan to mono float PCM. Returns empty if plan has no chords. */
std::vector<float> renderMidiToPcm(const ValidatedOutputPlan& outputPlan, double sampleRate);

} // namespace kelly
