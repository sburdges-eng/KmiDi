/**
 * Minimal KmiDi engine implementation — C ABI for RT harness and future plugin/FFI.
 * This is a stub that satisfies the ABI; real DSP can be wired in later.
 */
#include "engine.h"
#include <stdlib.h>
#include <string.h>

#define KMIDI_ENGINE_MAGIC 0x4B454E47u /* "KENG" */

struct kmidi_engine {
  uint32_t magic;
  uint32_t sample_rate;
  uint32_t block_size;
  uint32_t num_channels;
  int last_error_code;
};

static char s_last_error[256] = {0};

static void set_error(const char* msg) {
  (void)msg;
  s_last_error[0] = '\0';
  /* Could strncpy msg into s_last_error if we wanted to expose it. */
}

kmidi_engine_t* kmidi_engine_create(const kmidi_engine_config_t* config) {
  if (!config || config->sample_rate == 0 || config->max_block_size == 0 ||
      config->num_channels == 0) {
    set_error("invalid config");
    return NULL;
  }
  kmidi_engine_t* e = (kmidi_engine_t*)calloc(1, sizeof(kmidi_engine_t));
  if (!e) return NULL;
  e->magic = KMIDI_ENGINE_MAGIC;
  e->sample_rate = config->sample_rate;
  e->block_size = config->max_block_size;
  e->num_channels = config->num_channels;
  return e;
}

void kmidi_engine_destroy(kmidi_engine_t* engine) {
  if (!engine) return;
  engine->magic = 0;
  free(engine);
}

int kmidi_engine_prepare(kmidi_engine_t* engine,
                         uint32_t sample_rate,
                         uint32_t block_size) {
  if (!engine || engine->magic != KMIDI_ENGINE_MAGIC) return -1;
  if (sample_rate == 0 || block_size == 0) return -1;
  if (block_size > engine->block_size) return -1;
  engine->sample_rate = sample_rate;
  engine->block_size = block_size;
  return 0;
}

int kmidi_engine_process(kmidi_engine_t* engine,
                         float* const* audio,
                         uint32_t num_channels,
                         uint32_t num_samples) {
  if (!engine || engine->magic != KMIDI_ENGINE_MAGIC) return -1;
  if (!audio || num_channels != engine->num_channels ||
      num_samples > engine->block_size) {
    return -1;
  }
  /* Stub: no DSP yet; just touch buffers so they are not optimized away. */
  for (uint32_t c = 0; c < num_channels; c++) {
    if (audio[c]) {
      volatile float* p = audio[c];
      (void)p;
      (void)num_samples;
    }
  }
  return 0;
}

const char* kmidi_engine_last_error(void) {
  return s_last_error[0] ? s_last_error : NULL;
}

size_t kmidi_engine_query_scratch(const kmidi_engine_t* engine) {
  (void)engine;
  return 0;
}
