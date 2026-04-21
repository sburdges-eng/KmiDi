/**
 * Engine real-time state extension (Phase 3).
 *
 * This is a C++ file (not .c) because it uses penta::RTState (atomics)
 * and moodycamel::ReaderWriterQueue. It implements the C ABI functions
 * declared in engine.h.
 */
#include "engine.h"
#include "penta/common/RTState.h"
#include "readerwriterqueue.h"

#include <cstring>

// The pure-C engine struct is defined in engine.c.
// We extend it here with an RT companion that is lazily attached.
// For now, we use a static singleton — one engine per process.
// TODO: embed RTState into kmidi_engine struct when engine.c becomes C++.

static penta::RTState g_rtState;
static moodycamel::ReaderWriterQueue<penta::RTParameterUpdate> g_rtParamQueue(256);

extern "C" {

int kmidi_engine_get_state(const kmidi_engine_t* engine,
                           kmidi_engine_state_t* out_state) {
    if (!engine || !out_state) return -1;

    std::memset(out_state, 0, sizeof(*out_state));

    // Seqlock snapshot (shared helper from RTState.h). Retries up to 8x
    // on odd sequence or torn window; returns -1 (retry) on exhaustion.
    const bool ok = penta::rt_state_snapshot(g_rtState, [&]() noexcept {
        out_state->bpm             = g_rtState.bpm.load(std::memory_order_relaxed);
        out_state->sample_position = g_rtState.samplePosition.load(std::memory_order_relaxed);
        out_state->bar             = g_rtState.bar.load(std::memory_order_relaxed);
        out_state->beat            = g_rtState.beat.load(std::memory_order_relaxed);
        out_state->numerator       = g_rtState.numerator.load(std::memory_order_relaxed);
        out_state->denominator     = g_rtState.denominator.load(std::memory_order_relaxed);
        out_state->playing         = g_rtState.playing.load(std::memory_order_relaxed) ? 1 : 0;

        out_state->valence           = g_rtState.valence.load(std::memory_order_relaxed);
        out_state->arousal           = g_rtState.arousal.load(std::memory_order_relaxed);
        out_state->dominance         = g_rtState.dominance.load(std::memory_order_relaxed);
        out_state->emotion_intensity = g_rtState.emotionIntensity.load(std::memory_order_relaxed);

        out_state->groove_strength   = g_rtState.grooveStrength.load(std::memory_order_relaxed);
        out_state->harmonic_tension  = g_rtState.harmonicTension.load(std::memory_order_relaxed);
        out_state->rhythmic_density  = g_rtState.rhythmicDensity.load(std::memory_order_relaxed);
        out_state->melodic_activity  = g_rtState.melodicActivity.load(std::memory_order_relaxed);
    });

    if (!ok) return -1;

    out_state->sequence = g_rtState.sequence.load(std::memory_order_relaxed);
    return 0;
}

int kmidi_engine_push_param(kmidi_engine_t* engine,
                            kmidi_param_target_t target,
                            uint8_t param_index,
                            float value) {
    if (!engine) return -1;

    // Validate enum range
    if (target < KMIDI_PARAM_BPM || target > KMIDI_PARAM_TRANSPORT) return -1;

    // Validate track param index (Target::TrackParam == 3, maps to KMIDI_PARAM_TRANSPORT-1)
    // Note: engine.h doesn't expose TrackParam separately, but defend anyway
    if (param_index >= penta::kMaxTrackParams) return -1;

    penta::RTParameterUpdate update(
        static_cast<penta::RTParameterUpdate::Target>(target),
        param_index,
        value,
        0  // immediate
    );

    return g_rtParamQueue.try_enqueue(update) ? 0 : -1;
}

} // extern "C"
