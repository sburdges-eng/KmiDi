#include "kelly_ffi.h"
#include "common/KellyTypes.h"
#include "engine/EmotionThesaurus.h"
#include "engine/KellyBrain.h"
#include "penta/common/RTState.h"
#include "readerwriterqueue.h"
#include <atomic>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

// =============================================================================
// Internal Structures and Utilities
// =============================================================================

/**
 * Internal wrapper for KellyBrain C++ object
 */
// RT parameter update queue capacity
constexpr size_t kRTParamQueueCapacity = 256;

struct KellyBrainWrapper {
  std::unique_ptr<kelly::KellyBrain> brain;
  KellyEventCallback callback;
  void *callback_user_data;
  std::mutex mutex; // Thread safety for state access
  std::atomic<bool> initialized;

  // Phase 3: Real-time state and parameter queue
  penta::RTState rtState;
  moodycamel::ReaderWriterQueue<penta::RTParameterUpdate> rtParamQueue;

  KellyBrainWrapper()
      : callback(nullptr), callback_user_data(nullptr), initialized(false),
        rtParamQueue(kRTParamQueueCapacity) {
    brain = std::make_unique<kelly::KellyBrain>();
  }
};

using MallocString = std::unique_ptr<char, decltype(&std::free)>;

static KellyBrainWrapper *as_wrapper(KellyBrain *brain) noexcept {
  return reinterpret_cast<KellyBrainWrapper *>(brain);
}

static const KellyBrainWrapper *as_wrapper(const KellyBrain *brain) noexcept {
  return reinterpret_cast<const KellyBrainWrapper *>(brain);
}

// Thread-local fixed-size error buffer (RT-safe: no heap allocation)
static thread_local char tl_error_buf[512] = {};

/**
 * Set thread-local error message (RT-safe, no heap alloc)
 */
static void set_last_error(const char *msg) {
  std::strncpy(tl_error_buf, msg ? msg : "", sizeof(tl_error_buf) - 1);
  tl_error_buf[sizeof(tl_error_buf) - 1] = '\0';
}

static void set_last_error(const std::string &message) {
  set_last_error(message.c_str());
}

/**
 * Convert C++ string to C string (caller must free with kelly_free_string)
 */
static char *string_to_c_str(const std::string &str) {
  if (str.empty()) {
    return nullptr;
  }

  size_t len = str.length() + 1;
  MallocString result(static_cast<char *>(std::malloc(len)), &std::free);
  if (result == nullptr) {
    set_last_error("Memory allocation failed");
    return nullptr;
  }

  std::memcpy(result.get(), str.c_str(), len);
  return result.release();
}

/**
 * Escape a string for safe embedding in a JSON string value.
 * Handles backslash, double-quote, and ASCII control characters.
 */
static std::string escape_json_string(const std::string &s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
    case '\\':
      out += "\\\\";
      break;
    case '"':
      out += "\\\"";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    case '\b':
      out += "\\b";
      break;
    case '\f':
      out += "\\f";
      break;
    default:
      if (static_cast<unsigned char>(c) < 0x20) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "\\u%04x",
                      static_cast<unsigned>(static_cast<unsigned char>(c)));
        out += buf;
      } else {
        out += c;
      }
      break;
    }
  }
  return out;
}

static std::string serialize_intent_result(const kelly::IntentResult &result) {
  std::ostringstream json;
  json << "{\n";

  // Extract wound information from sourceWound
  json << "  \"core_wound\": \""
       << escape_json_string(result.sourceWound.description) << "\",\n";
  json << "  \"core_desire\": \""
       << escape_json_string(result.sourceWound.desire) << "\",\n";
  json << "  \"emotional_intent\": \""
       << escape_json_string(result.sourceWound.expression) << "\",\n";

  // Extract emotion values from sourceWound.primaryEmotion
  json << "  \"valence\": " << result.sourceWound.primaryEmotion.valence
       << ",\n";
  json << "  \"arousal\": " << result.sourceWound.primaryEmotion.arousal
       << ",\n";
  json << "  \"dominance\": " << result.sourceWound.primaryEmotion.dominance
       << ",\n";
  json << "  \"complexity\": "
       << result.sourceWound.primaryEmotion.musicalAttributes.density << ",\n";

  // Musical parameters from IntentResult
  json << "  \"progression\": [";
  for (size_t i = 0; i < result.chordProgression.size(); ++i) {
    if (i > 0)
      json << ", ";
    json << "\"" << escape_json_string(result.chordProgression[i]) << "\"";
  }
  json << "],\n";

  json << "  \"key\": \"" << escape_json_string(result.key) << "\",\n";
  json << "  \"bpm\": " << result.tempoBpm << ",\n";
  json << "  \"mode\": \"" << escape_json_string(result.mode) << "\",\n";
  json << "  \"time_signature\": \"" << result.timeSignature.numerator << "/"
       << result.timeSignature.denominator << "\",\n";
  json << "  \"confidence\": " << result.confidence << ",\n";

  // Melodic guidance (theoretical parameters)
  json << "  \"melodic_range\": " << result.melodicRange << ",\n";
  json << "  \"leap_probability\": " << result.leapProbability << ",\n";
  json << "  \"allow_chromaticism\": "
       << (result.allowChromaticism ? "true" : "false") << ",\n";
  json << "  \"allow_dissonance\": "
       << (result.allowDissonance ? "true" : "false") << ",\n";

  // Rhythmic guidance (theoretical parameters)
  json << "  \"swing_amount\": " << result.swingAmount << ",\n";
  json << "  \"syncopation_level\": " << result.syncopationLevel << ",\n";
  json << "  \"humanization\": " << result.humanization << ",\n";

  // Dynamics (theoretical parameters)
  json << "  \"base_velocity\": " << result.baseVelocity << ",\n";
  json << "  \"dynamic_range\": " << result.dynamicRange << ",\n";

  // Production notes (theoretical parameters)
  json << "  \"production_notes\": [";
  for (size_t i = 0; i < result.productionNotes.size(); ++i) {
    if (i > 0)
      json << ", ";
    json << "\"" << escape_json_string(result.productionNotes[i]) << "\"";
  }
  json << "],\n";
  json << "  \"narrative_arc\": \"" << escape_json_string(result.narrativeArc)
       << "\",\n";

  // Rule breaks (theoretical parameters)
  json << "  \"rule_breaks\": [";
  for (size_t i = 0; i < result.ruleBreaks.size(); ++i) {
    if (i > 0)
      json << ", ";
    const auto &rb = result.ruleBreaks[i];
    json << "{\"type\": " << static_cast<int>(rb.type)
         << ", \"description\": \"" << escape_json_string(rb.description)
         << "\", \"justification\": \"" << escape_json_string(rb.justification)
         << "\", \"intensity\": " << rb.intensity << "}";
  }
  json << "]\n";

  json << "}";

  return json.str();
}

/**
 * Simple JSON serialization for MIDI data
 */
static std::string serialize_generated_midi(const kelly::GeneratedMidi &midi) {
  std::ostringstream json;
  json << "{\n";
  json << "  \"bars\": " << midi.bars << ",\n";
  json << "  \"bpm\": " << midi.tempoBpm << ",\n";
  json << "  \"key\": \"" << escape_json_string(midi.key) << "\",\n";
  json << "  \"mode\": \"" << escape_json_string(midi.mode) << "\",\n";

  // Serialize notes grouped by channel (simulating tracks)
  std::map<int, std::vector<kelly::MidiNote>> notesByChannel;
  for (const auto &note : midi.notes) {
    notesByChannel[note.channel].push_back(note);
  }

  json << "  \"tracks\": [\n";
  bool first_track = true;
  for (const auto &[channel, notes] : notesByChannel) {
    if (!first_track)
      json << ",\n";
    first_track = false;

    json << "    {\n";
    json << "      \"name\": \"Channel " << channel << "\",\n";
    json << "      \"channel\": " << channel << ",\n";
    json << "      \"events\": [\n";

    for (size_t j = 0; j < notes.size(); ++j) {
      if (j > 0)
        json << ",\n";
      const auto &note = notes[j];
      json << "        {\n";
      json << "          \"event_type\": 1,\n"; // Note on
      json << "          \"timestamp\": " << (note.startTick / 480.0) << ",\n";
      json << "          \"note\": " << note.pitch << ",\n";
      json << "          \"velocity\": " << note.velocity << ",\n";
      json << "          \"duration\": " << (note.durationTicks / 480.0)
           << "\n";
      json << "        }";
    }

    json << "\n      ]\n";
    json << "    }";
  }

  json << "\n  ],\n";
  json << "  \"chords\": [";
  for (size_t i = 0; i < midi.chords.size(); ++i) {
    if (i > 0)
      json << ", ";
    json << "\"" << escape_json_string(midi.chords[i].symbol) << "\"";
  }
  json << "]\n";
  json << "}";

  return json.str();
}

/**
 * Parse JSON for emotion parameters (simplified parser)
 */
static kelly::Wound parse_wound_json(const std::string &json) {
  kelly::Wound wound;

  // Very basic JSON parsing - in production, use a proper JSON library
  // Extract description
  size_t desc_pos = json.find("\"description\":");
  if (desc_pos != std::string::npos) {
    size_t start = json.find("\"", desc_pos + 14) + 1;
    size_t end = json.find("\"", start);
    if (start < end) {
      wound.description = json.substr(start, end - start);
    }
  }

  // Extract desire
  size_t desire_pos = json.find("\"desire\":");
  if (desire_pos != std::string::npos) {
    size_t start = json.find("\"", desire_pos + 10) + 1;
    size_t end = json.find("\"", start);
    if (start < end) {
      wound.desire = json.substr(start, end - start);
    }
  }

  // Extract expression
  size_t expr_pos = json.find("\"expression\":");
  if (expr_pos != std::string::npos) {
    size_t start = json.find("\"", expr_pos + 13) + 1;
    size_t end = json.find("\"", start);
    if (start < end) {
      wound.expression = json.substr(start, end - start);
    }
  }

  // Extract urgency (prefer explicit "urgency" key)
  size_t urgency_pos = json.find("\"urgency\":");
  if (urgency_pos != std::string::npos) {
    size_t colon = urgency_pos + 10;
    size_t start = json.find_first_not_of(" \t", colon);
    if (start != std::string::npos) {
      try {
        wound.urgency = std::stof(json.substr(start));
      } catch (...) {
        wound.urgency = 0.5f;
      }
    }
  }

  // Extract intensity (prefer explicit "intensity" key; fall back to urgency)
  size_t intensity_pos = json.find("\"intensity\":");
  if (intensity_pos != std::string::npos) {
    size_t colon = intensity_pos + 12;
    size_t start = json.find_first_not_of(" \t", colon);
    if (start != std::string::npos) {
      try {
        wound.intensity = std::stof(json.substr(start));
      } catch (...) {
        wound.intensity = 0.5f;
      }
    }
  }

  // Keep urgency and intensity in sync when only one is provided
  if (urgency_pos != std::string::npos && intensity_pos == std::string::npos) {
    wound.intensity = wound.urgency;
  } else if (intensity_pos != std::string::npos &&
             urgency_pos == std::string::npos) {
    wound.urgency = wound.intensity;
  }

  return wound;
}

// =============================================================================
// Error Handling Functions
// =============================================================================

extern "C" {

const char *kelly_get_error_message(KellyErrorCode error_code) {
  switch (error_code) {
  case KELLY_SUCCESS:
    return "Success";
  case KELLY_ERROR_NULL_POINTER:
    return "Null pointer error";
  case KELLY_ERROR_INITIALIZATION_FAILED:
    return "Initialization failed";
  case KELLY_ERROR_INVALID_PARAMETER:
    return "Invalid parameter";
  case KELLY_ERROR_JSON_PARSE_ERROR:
    return "JSON parse error";
  case KELLY_ERROR_MEMORY_ALLOCATION:
    return "Memory allocation failed";
  case KELLY_ERROR_FILE_NOT_FOUND:
    return "File not found";
  case KELLY_ERROR_AGAIN:
    return "Transient seqlock contention; retry the call";
  case KELLY_ERROR_UNKNOWN:
  default:
    return "Unknown error";
  }
}

const char *kelly_get_last_error(void) {
  return tl_error_buf[0] != '\0' ? tl_error_buf : nullptr;
}

// =============================================================================
// Memory Management Functions
// =============================================================================

void kelly_free_string(char *ptr) {
  if (ptr != nullptr) {
    std::free(ptr);
  }
}

// =============================================================================
// KellyBrain Core Functions
// =============================================================================

KellyBrain *kelly_brain_create(void) {
  try {
    auto wrapper = std::make_unique<KellyBrainWrapper>();
    return reinterpret_cast<KellyBrain *>(wrapper.release());
  } catch (const std::exception &e) {
    set_last_error(std::string("Exception in kelly_brain_create: ") + e.what());
    return nullptr;
  }
}

KellyErrorCode kelly_brain_initialize(KellyBrain *brain,
                                      const char *data_path) {
  if (brain == nullptr || data_path == nullptr) {
    set_last_error("Null pointer in kelly_brain_initialize");
    return KELLY_ERROR_NULL_POINTER;
  }

  try {
    auto *wrapper = as_wrapper(brain);
    std::lock_guard<std::mutex> lock(wrapper->mutex);

    bool success = wrapper->brain->initialize(std::string(data_path));
    if (success) {
      wrapper->initialized = true;
      return KELLY_SUCCESS;
    } else {
      set_last_error("KellyBrain initialization failed");
      return KELLY_ERROR_INITIALIZATION_FAILED;
    }
  } catch (const std::exception &e) {
    set_last_error(std::string("Exception in kelly_brain_initialize: ") +
                   e.what());
    return KELLY_ERROR_INITIALIZATION_FAILED;
  }
}

bool kelly_brain_is_initialized(const KellyBrain *brain) {
  if (brain == nullptr) {
    return false;
  }

  auto *wrapper = as_wrapper(brain);
  return wrapper->initialized;
}

void kelly_brain_destroy(KellyBrain *brain) {
  if (brain != nullptr) {
    std::unique_ptr<KellyBrainWrapper> wrapper(as_wrapper(brain));
  }
}

// =============================================================================
// Intent Processing Functions
// =============================================================================

char *kelly_brain_from_text(KellyBrain *brain, const char *text) {
  if (brain == nullptr || text == nullptr) {
    set_last_error("Null pointer in kelly_brain_from_text");
    return nullptr;
  }

  try {
    auto *wrapper = as_wrapper(brain);
    std::lock_guard<std::mutex> lock(wrapper->mutex);

    if (!wrapper->initialized) {
      set_last_error("KellyBrain not initialized");
      return nullptr;
    }

    kelly::IntentResult result = wrapper->brain->fromText(std::string(text));
    std::string json = serialize_intent_result(result);

    return string_to_c_str(json);
  } catch (const std::exception &e) {
    set_last_error(std::string("Exception in kelly_brain_from_text: ") +
                   e.what());
    return nullptr;
  }
}

char *kelly_brain_from_emotion(KellyBrain *brain, const char *emotion_name,
                               float intensity) {
  if (brain == nullptr || emotion_name == nullptr) {
    set_last_error("Null pointer in kelly_brain_from_emotion");
    return nullptr;
  }

  if (intensity < 0.0f || intensity > 1.0f) {
    set_last_error("Invalid intensity value, must be 0.0 to 1.0");
    return nullptr;
  }

  try {
    auto *wrapper = as_wrapper(brain);
    std::lock_guard<std::mutex> lock(wrapper->mutex);

    if (!wrapper->initialized) {
      set_last_error("KellyBrain not initialized");
      return nullptr;
    }

    kelly::IntentResult result =
        wrapper->brain->fromEmotion(std::string(emotion_name), intensity);
    std::string json = serialize_intent_result(result);

    return string_to_c_str(json);
  } catch (const std::exception &e) {
    set_last_error(std::string("Exception in kelly_brain_from_emotion: ") +
                   e.what());
    return nullptr;
  }
}

char *kelly_brain_from_journey(KellyBrain *brain,
                               const char *current_state_json,
                               const char *desired_state_json) {
  if (brain == nullptr || current_state_json == nullptr ||
      desired_state_json == nullptr) {
    set_last_error("Null pointer in kelly_brain_from_journey");
    return nullptr;
  }

  try {
    auto *wrapper = as_wrapper(brain);
    std::lock_guard<std::mutex> lock(wrapper->mutex);

    if (!wrapper->initialized) {
      set_last_error("KellyBrain not initialized");
      return nullptr;
    }

    // Parse JSON to SideA/SideB structures (simplified for now)
    kelly::Wound w = parse_wound_json(std::string(current_state_json));
    kelly::SideA current;
    current.description = w.description;
    current.intensity = w.intensity;
    if (w.primaryEmotion.id >= 0)
      current.emotionId = w.primaryEmotion.id;
    kelly::SideB desired;
    // desired parsing would be similar

    kelly::IntentResult result = wrapper->brain->fromJourney(current, desired);
    std::string json = serialize_intent_result(result);

    return string_to_c_str(json);
  } catch (const std::exception &e) {
    set_last_error(std::string("Exception in kelly_brain_from_journey: ") +
                   e.what());
    return nullptr;
  }
}

char *kelly_brain_from_wound(KellyBrain *brain, const char *wound_json) {
  if (brain == nullptr || wound_json == nullptr) {
    set_last_error("Null pointer in kelly_brain_from_wound");
    return nullptr;
  }

  try {
    auto *wrapper = as_wrapper(brain);
    std::lock_guard<std::mutex> lock(wrapper->mutex);

    if (!wrapper->initialized) {
      set_last_error("KellyBrain not initialized");
      return nullptr;
    }

    kelly::Wound wound = parse_wound_json(std::string(wound_json));
    kelly::IntentResult result = wrapper->brain->fromWound(wound);
    std::string json = serialize_intent_result(result);

    return string_to_c_str(json);
  } catch (const std::exception &e) {
    set_last_error(std::string("Exception in kelly_brain_from_wound: ") +
                   e.what());
    return nullptr;
  }
}

// =============================================================================
// MIDI Generation Functions
// =============================================================================

char *kelly_brain_generate_midi(KellyBrain *brain, const char *intent_json,
                                int bars) {
  if (brain == nullptr || intent_json == nullptr) {
    set_last_error("Null pointer in kelly_brain_generate_midi");
    return nullptr;
  }

  if (bars <= 0 || bars > 64) {
    set_last_error("Invalid bars value, must be 1-64");
    return nullptr;
  }

  try {
    auto *wrapper = as_wrapper(brain);
    std::lock_guard<std::mutex> lock(wrapper->mutex);

    if (!wrapper->initialized) {
      set_last_error("KellyBrain not initialized");
      return nullptr;
    }

    // Parse intent JSON back to IntentResult (simplified)
    kelly::IntentResult intent;
    // In production, use proper JSON parsing

    kelly::GeneratedMidi midi = wrapper->brain->generateMidi(intent, bars);
    std::string json = serialize_generated_midi(midi);

    return string_to_c_str(json);
  } catch (const std::exception &e) {
    set_last_error(std::string("Exception in kelly_brain_generate_midi: ") +
                   e.what());
    return nullptr;
  }
}

char *kelly_brain_generate_midi_with_params(KellyBrain *brain,
                                            const char *intent_json, int bars,
                                            int bpm,
                                            const char *key_signature) {
  if (brain == nullptr || intent_json == nullptr || key_signature == nullptr) {
    set_last_error("Null pointer in kelly_brain_generate_midi_with_params");
    return nullptr;
  }

  if (bars <= 0 || bars > 64 || bpm <= 0 || bpm > 300) {
    set_last_error("Invalid parameters");
    return nullptr;
  }

  try {
    auto *wrapper = as_wrapper(brain);
    std::lock_guard<std::mutex> lock(wrapper->mutex);

    if (!wrapper->initialized) {
      set_last_error("KellyBrain not initialized");
      return nullptr;
    }

    // Parse intent JSON and generate with specific parameters
    kelly::IntentResult intent;
    // Set specific BPM and key
    intent.tempoBpm = bpm;
    intent.key = std::string(key_signature);

    kelly::GeneratedMidi midi = wrapper->brain->generateMidi(intent, bars);
    std::string json = serialize_generated_midi(midi);

    return string_to_c_str(json);
  } catch (const std::exception &e) {
    set_last_error(
        std::string("Exception in kelly_brain_generate_midi_with_params: ") +
        e.what());
    return nullptr;
  }
}

// =============================================================================
// State Query Functions
// =============================================================================

char *kelly_brain_get_emotion_state(const KellyBrain *brain) {
  if (brain == nullptr) {
    set_last_error("Null pointer in kelly_brain_get_emotion_state");
    return nullptr;
  }

  try {
    auto *wrapper = as_wrapper(brain);

    if (!wrapper->initialized) {
      set_last_error("KellyBrain not initialized");
      return nullptr;
    }

    // Get current emotion state (simplified)
    std::ostringstream json;
    json << "{\n";
    json << "  \"valence\": 0.0,\n";
    json << "  \"arousal\": 0.5,\n";
    json << "  \"dominance\": 0.5,\n";
    json << "  \"complexity\": 0.3\n";
    json << "}";

    return string_to_c_str(json.str());
  } catch (const std::exception &e) {
    set_last_error(std::string("Exception in kelly_brain_get_emotion_state: ") +
                   e.what());
    return nullptr;
  }
}

char *kelly_brain_get_parameters(const KellyBrain *brain) {
  if (brain == nullptr) {
    set_last_error("Null pointer in kelly_brain_get_parameters");
    return nullptr;
  }

  try {
    // Return current parameters as JSON
    std::ostringstream json;
    json << "{\n";
    json << "  \"processing_enabled\": true,\n";
    json << "  \"default_bpm\": 120,\n";
    json << "  \"default_key\": \"C\",\n";
    json << "  \"complexity_factor\": 0.7\n";
    json << "}";

    return string_to_c_str(json.str());
  } catch (const std::exception &e) {
    set_last_error(std::string("Exception in kelly_brain_get_parameters: ") +
                   e.what());
    return nullptr;
  }
}

char *kelly_brain_get_available_emotions(const KellyBrain *brain) {
  if (brain == nullptr) {
    set_last_error("Null pointer in kelly_brain_get_available_emotions");
    return nullptr;
  }

  try {
    auto *wrapper = as_wrapper(brain);

    if (!wrapper->initialized) {
      set_last_error("KellyBrain not initialized");
      return nullptr;
    }

    // Return available emotions (would come from EmotionThesaurus)
    std::ostringstream json;
    json << "{\n";
    json << "  \"emotions\": [\n";
    json << "    {\"name\": \"joy\", \"category\": \"positive\"},\n";
    json << "    {\"name\": \"sadness\", \"category\": \"negative\"},\n";
    json << "    {\"name\": \"anger\", \"category\": \"negative\"},\n";
    json << "    {\"name\": \"fear\", \"category\": \"negative\"},\n";
    json << "    {\"name\": \"love\", \"category\": \"positive\"},\n";
    json << "    {\"name\": \"hope\", \"category\": \"positive\"}\n";
    json << "  ]\n";
    json << "}";

    return string_to_c_str(json.str());
  } catch (const std::exception &e) {
    set_last_error(
        std::string("Exception in kelly_brain_get_available_emotions: ") +
        e.what());
    return nullptr;
  }
}

// =============================================================================
// Parameter Update Functions
// =============================================================================

KellyErrorCode kelly_brain_set_emotion_parameters(KellyBrain *brain,
                                                  float valence, float arousal,
                                                  float dominance) {
  if (brain == nullptr) {
    set_last_error("Null pointer in kelly_brain_set_emotion_parameters");
    return KELLY_ERROR_NULL_POINTER;
  }

  if (valence < -1.0f || valence > 1.0f || arousal < 0.0f || arousal > 1.0f ||
      dominance < 0.0f || dominance > 1.0f) {
    set_last_error("Invalid parameter values");
    return KELLY_ERROR_INVALID_PARAMETER;
  }

  try {
    auto *wrapper = as_wrapper(brain);

    // Copy callback and build event data under the lock, then invoke outside
    // to prevent deadlock if the callback re-enters any kelly_brain_* function.
    decltype(wrapper->callback) cb = nullptr;
    void *cb_user_data = nullptr;
    std::string event_data_str;

    {
      std::lock_guard<std::mutex> lock(wrapper->mutex);

      if (!wrapper->initialized) {
        set_last_error("KellyBrain not initialized");
        return KELLY_ERROR_INITIALIZATION_FAILED;
      }

      // Update emotional parameters
      // In the actual implementation, this would update the internal state

      cb = wrapper->callback;
      cb_user_data = wrapper->callback_user_data;
      if (cb != nullptr) {
        std::ostringstream event_data;
        event_data << "{\"valence\": " << valence
                   << ", \"arousal\": " << arousal
                   << ", \"dominance\": " << dominance << "}";
        event_data_str = event_data.str();
      }
    } // lock released here

    if (cb != nullptr) {
      cb("emotion_changed", event_data_str.c_str(), cb_user_data);
    }

    return KELLY_SUCCESS;
  } catch (const std::exception &e) {
    set_last_error(
        std::string("Exception in kelly_brain_set_emotion_parameters: ") +
        e.what());
    return KELLY_ERROR_UNKNOWN;
  }
}

KellyErrorCode kelly_brain_update_parameters(KellyBrain *brain,
                                             const char *params_json) {
  if (brain == nullptr || params_json == nullptr) {
    set_last_error("Null pointer in kelly_brain_update_parameters");
    return KELLY_ERROR_NULL_POINTER;
  }

  try {
    auto *wrapper = as_wrapper(brain);
    std::lock_guard<std::mutex> lock(wrapper->mutex);

    if (!wrapper->initialized) {
      set_last_error("KellyBrain not initialized");
      return KELLY_ERROR_INITIALIZATION_FAILED;
    }

    // Parse and update parameters
    // In production, use proper JSON parsing

    return KELLY_SUCCESS;
  } catch (const std::exception &e) {
    set_last_error(std::string("Exception in kelly_brain_update_parameters: ") +
                   e.what());
    return KELLY_ERROR_UNKNOWN;
  }
}

// =============================================================================
// Event/Callback System
// =============================================================================

KellyErrorCode kelly_brain_register_callback(KellyBrain *brain,
                                             KellyEventCallback callback,
                                             void *user_data) {
  if (brain == nullptr || callback == nullptr) {
    set_last_error("Null pointer in kelly_brain_register_callback");
    return KELLY_ERROR_NULL_POINTER;
  }

  try {
    auto *wrapper = as_wrapper(brain);
    std::lock_guard<std::mutex> lock(wrapper->mutex);

    wrapper->callback = callback;
    wrapper->callback_user_data = user_data;

    return KELLY_SUCCESS;
  } catch (const std::exception &e) {
    set_last_error(std::string("Exception in kelly_brain_register_callback: ") +
                   e.what());
    return KELLY_ERROR_UNKNOWN;
  }
}

KellyErrorCode kelly_brain_unregister_callback(KellyBrain *brain) {
  if (brain == nullptr) {
    set_last_error("Null pointer in kelly_brain_unregister_callback");
    return KELLY_ERROR_NULL_POINTER;
  }

  try {
    auto *wrapper = as_wrapper(brain);
    std::lock_guard<std::mutex> lock(wrapper->mutex);

    wrapper->callback = nullptr;
    wrapper->callback_user_data = nullptr;

    return KELLY_SUCCESS;
  } catch (const std::exception &e) {
    set_last_error(
        std::string("Exception in kelly_brain_unregister_callback: ") +
        e.what());
    return KELLY_ERROR_UNKNOWN;
  }
}

// =============================================================================
// Utility Functions
// =============================================================================

const char *kelly_get_version(void) { return "KellyBrain FFI v1.0.0"; }

bool kelly_check_data_files(const char *data_path) {
  if (data_path == nullptr) {
    return false;
  }

  // Check if required data files exist
  // This would check for emotion thesaurus files, etc.
  // Simplified implementation for now
  return true;
}

// =============================================================================
// Real-Time State Interface (Phase 3)
// =============================================================================

KellyErrorCode kelly_brain_get_rt_state(const KellyBrain *brain,
                                        KellyRTState *out_state) {
  if (brain == nullptr || out_state == nullptr) {
    return KELLY_ERROR_NULL_POINTER;
  }

  auto *wrapper = as_wrapper(brain);
  const auto &s = wrapper->rtState;

  static_assert(sizeof(out_state->track_params) / sizeof(float) ==
                    penta::kMaxTrackParams,
                "KellyRTState.track_params size must match kMaxTrackParams");

  // Seqlock snapshot: if the audio thread is mid-publish (odd sequence) or
  // bumps the sequence between our two reads (torn window), retry up to 64x.
  // On exhaustion, surface KELLY_ERROR_AGAIN so the caller can retry rather
  // than consume half-old / half-new state.
  //
  // captured_seq is set inside the lambda to the seq1 value observed for the
  // successful round. This avoids a race where the writer advances between
  // the snapshot return and a post-snapshot sequence reload.
  uint64_t captured_seq = 0;
  const bool ok = penta::rt_state_snapshot(s, [&]() noexcept {
    // Capture the current (even, stable) sequence inside the window.
    captured_seq = s.sequence.load(std::memory_order_relaxed);

    out_state->bpm = s.bpm.load(std::memory_order_relaxed);
    out_state->sample_position =
        s.samplePosition.load(std::memory_order_relaxed);
    out_state->bar_start = s.barStart.load(std::memory_order_relaxed);
    out_state->bar = s.bar.load(std::memory_order_relaxed);
    out_state->beat = s.beat.load(std::memory_order_relaxed);
    out_state->numerator = s.numerator.load(std::memory_order_relaxed);
    out_state->denominator = s.denominator.load(std::memory_order_relaxed);
    out_state->playing = s.playing.load(std::memory_order_relaxed) ? 1 : 0;

    out_state->valence = s.valence.load(std::memory_order_relaxed);
    out_state->arousal = s.arousal.load(std::memory_order_relaxed);
    out_state->dominance = s.dominance.load(std::memory_order_relaxed);
    out_state->discrete_emotion_id =
        s.discreteEmotionId.load(std::memory_order_relaxed);
    out_state->emotion_intensity =
        s.emotionIntensity.load(std::memory_order_relaxed);
    out_state->emotion_confidence =
        s.emotionConfidence.load(std::memory_order_relaxed);

    out_state->tempo_bias = s.tempoBias.load(std::memory_order_relaxed);
    out_state->rhythmic_density =
        s.rhythmicDensity.load(std::memory_order_relaxed);
    out_state->groove_strength =
        s.grooveStrength.load(std::memory_order_relaxed);
    out_state->harmonic_tension =
        s.harmonicTension.load(std::memory_order_relaxed);
    out_state->harmonic_motion =
        s.harmonicMotion.load(std::memory_order_relaxed);
    out_state->melodic_activity =
        s.melodicActivity.load(std::memory_order_relaxed);
    out_state->texture_density =
        s.textureDensity.load(std::memory_order_relaxed);
    out_state->dynamic_range = s.dynamicRange.load(std::memory_order_relaxed);

    for (size_t i = 0; i < penta::kMaxTrackParams; ++i) {
      out_state->track_params[i] =
          s.trackParams[i].load(std::memory_order_relaxed);
    }
  });

  if (!ok) {
    return KELLY_ERROR_AGAIN;
  }

  // Publish the sequence value captured inside the stable window.
  // Using the value observed during the snapshot (not re-read here)
  // ensures the sequence matches the payload — no post-snapshot race.
  out_state->sequence = captured_seq;

  return KELLY_SUCCESS;
}

KellyErrorCode kelly_brain_push_rt_param(KellyBrain *brain,
                                         KellyRTTarget target,
                                         uint8_t param_index, float value) {
  if (brain == nullptr) {
    return KELLY_ERROR_NULL_POINTER;
  }

  // Validate enum range
  if (target < KELLY_RT_TARGET_BPM || target > KELLY_RT_TARGET_TRANSPORT) {
    return KELLY_ERROR_INVALID_PARAMETER;
  }

  // Validate track param index bounds
  if (target == KELLY_RT_TARGET_TRACK_PARAM &&
      param_index >= penta::kMaxTrackParams) {
    return KELLY_ERROR_INVALID_PARAMETER;
  }

  auto *wrapper = as_wrapper(brain);

  penta::RTParameterUpdate update(
      static_cast<penta::RTParameterUpdate::Target>(target), param_index, value,
      0 // immediate
  );

  // try_enqueue is non-blocking; avoid set_last_error() here as this
  // may be called from a near-RT context (no heap allocs on this path).
  if (!wrapper->rtParamQueue.try_enqueue(update)) {
    return KELLY_ERROR_INVALID_PARAMETER;
  }

  return KELLY_SUCCESS;
}

} // extern "C"