/**
 * KmiDi engine C ABI — single entry point for DSP/latent engine.
 * See docs/research/MULTIMODAL_IMPLEMENTATIONS_PLAN.md section 3.1.
 */
#ifndef KMIDI_ENGINE_H
#define KMIDI_ENGINE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/** Opaque engine instance. */
typedef struct kmidi_engine kmidi_engine_t;

/** Configuration for engine creation. */
typedef struct kmidi_engine_config {
  /** Sample rate (e.g. 48000). */
  uint32_t sample_rate;
  /** Max block size (samples per channel per process call). */
  uint32_t max_block_size;
  /** Number of audio channels (e.g. 2). */
  uint32_t num_channels;
  /** Reserved for future use; set to 0. */
  uint32_t reserved;
} kmidi_engine_config_t;

/**
 * Create engine with the given config. Returns NULL on failure.
 */
kmidi_engine_t* kmidi_engine_create(const kmidi_engine_config_t* config);

/**
 * Destroy engine and free resources.
 */
void kmidi_engine_destroy(kmidi_engine_t* engine);

/**
 * Prepare for processing (call after create, or when sample rate / block size change).
 * sample_rate and block_size must match or be within engine config.
 * Returns 0 on success, non-zero on error.
 */
int kmidi_engine_prepare(kmidi_engine_t* engine,
                         uint32_t sample_rate,
                         uint32_t block_size);

/**
 * Process one block of audio.
 * audio: array of channel pointers, each pointing to block_size floats.
 * num_channels: must match config.
 * num_samples: samples per channel (block size).
 * Returns 0 on success, non-zero on error.
 */
int kmidi_engine_process(kmidi_engine_t* engine,
                         float* const* audio,
                         uint32_t num_channels,
                         uint32_t num_samples);

/**
 * Optional: get last error string (thread-local). May return NULL.
 */
const char* kmidi_engine_last_error(void);

/**
 * Optional: query required scratch size in bytes (for pre-allocation). 0 if not used.
 */
size_t kmidi_engine_query_scratch(const kmidi_engine_t* engine);

/* =========================================================================
 * Real-Time State (Phase 3)
 * ========================================================================= */

/** C-compatible engine state snapshot. */
typedef struct kmidi_engine_state {
  double   bpm;
  uint64_t sample_position;
  uint32_t bar;
  uint32_t beat;
  uint32_t numerator;
  uint32_t denominator;
  int      playing;

  float    valence;
  float    arousal;
  float    dominance;
  float    emotion_intensity;

  float    groove_strength;
  float    harmonic_tension;
  float    rhythmic_density;
  float    melodic_activity;

  uint64_t sequence;
} kmidi_engine_state_t;

/**
 * Get a snapshot of the engine's real-time state.
 * Lock-free — safe to call from any thread at any frequency.
 * Returns 0 on success, non-zero on error.
 */
int kmidi_engine_get_state(const kmidi_engine_t* engine,
                           kmidi_engine_state_t* out_state);

/** Parameter update target. */
typedef enum {
  KMIDI_PARAM_BPM        = 0,
  KMIDI_PARAM_EMOTION    = 1,
  KMIDI_PARAM_INTENT     = 2,
  KMIDI_PARAM_TRANSPORT  = 3
} kmidi_param_target_t;

/**
 * Push a parameter update into the RT-safe queue.
 * Returns 0 on success, -1 if queue is full.
 */
int kmidi_engine_push_param(kmidi_engine_t* engine,
                            kmidi_param_target_t target,
                            uint8_t param_index,
                            float value);

#ifdef __cplusplus
}
#endif

#endif /* KMIDI_ENGINE_H */
