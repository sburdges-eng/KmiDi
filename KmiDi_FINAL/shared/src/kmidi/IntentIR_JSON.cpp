#include "kmidi/IntentIR_JSON.h"
#include "kmidi/IntentIR.h"
#include <cstring>
#include <cstdlib>

extern "C" {

#ifdef HAVE_CJSON
#include <cJSON.h>

char* intent_frame_to_json(const IntentFrame* frame) {
    if (!frame) {
        return nullptr;
    }

    cJSON* json = cJSON_CreateObject();
    if (!json) {
        return nullptr;
    }

    // Meta
    cJSON* meta = cJSON_CreateObject();
    cJSON_AddNumberToObject(meta, "ir_version", frame->meta.ir_version);
    cJSON_AddNumberToObject(meta, "intent_id", (double)frame->meta.intent_id);
    cJSON_AddNumberToObject(meta, "session_id", (double)frame->meta.session_id);
    cJSON_AddItemToObject(json, "meta", meta);

    // Emotion
    cJSON* emotion = cJSON_CreateObject();
    cJSON_AddNumberToObject(emotion, "valence", frame->emotion.valence);
    cJSON_AddNumberToObject(emotion, "arousal", frame->emotion.arousal);
    cJSON_AddNumberToObject(emotion, "dominance", frame->emotion.dominance);
    cJSON_AddNumberToObject(emotion, "discrete_id", frame->emotion.discrete_id);
    cJSON_AddNumberToObject(emotion, "intensity", frame->emotion.intensity);
    cJSON_AddNumberToObject(emotion, "confidence", frame->emotion.confidence);
    cJSON_AddItemToObject(json, "emotion", emotion);

    // Music
    cJSON* music = cJSON_CreateObject();
    cJSON_AddNumberToObject(music, "tempo_bias", frame->music.tempo_bias);
    cJSON_AddNumberToObject(music, "rhythmic_density", frame->music.rhythmic_density);
    cJSON_AddNumberToObject(music, "groove_strength", frame->music.groove_strength);
    cJSON_AddNumberToObject(music, "harmonic_tension", frame->music.harmonic_tension);
    cJSON_AddNumberToObject(music, "harmonic_motion", frame->music.harmonic_motion);
    cJSON_AddNumberToObject(music, "mode_preference", frame->music.mode_preference);
    cJSON_AddNumberToObject(music, "melodic_activity", frame->music.melodic_activity);
    cJSON_AddNumberToObject(music, "contour_variance", frame->music.contour_variance);
    cJSON_AddNumberToObject(music, "dynamic_range", frame->music.dynamic_range);
    cJSON_AddNumberToObject(music, "texture_density", frame->music.texture_density);
    cJSON_AddItemToObject(json, "music", music);

    // Time
    cJSON* time = cJSON_CreateObject();
    cJSON_AddNumberToObject(time, "start_bar", frame->time.start_bar);
    cJSON_AddNumberToObject(time, "end_bar", frame->time.end_bar);
    cJSON_AddNumberToObject(time, "fade_in_beats", frame->time.fade_in_beats);
    cJSON_AddNumberToObject(time, "fade_out_beats", frame->time.fade_out_beats);
    cJSON_AddItemToObject(json, "time", time);

    // Constraints
    cJSON* constraints = cJSON_CreateObject();
    cJSON_AddNumberToObject(constraints, "allowed_engines_mask", (double)frame->constraints.allowed_engines_mask);
    cJSON_AddNumberToObject(constraints, "forbidden_engines_mask", (double)frame->constraints.forbidden_engines_mask);
    cJSON_AddNumberToObject(constraints, "max_cpu_cost", frame->constraints.max_cpu_cost);
    cJSON_AddNumberToObject(constraints, "max_event_rate", frame->constraints.max_event_rate);
    cJSON_AddItemToObject(json, "constraints", constraints);

    // Provenance
    cJSON* provenance = cJSON_CreateObject();
    cJSON_AddNumberToObject(provenance, "source", frame->provenance.source);
    cJSON_AddNumberToObject(provenance, "user_override_weight", frame->provenance.user_override_weight);
    cJSON_AddItemToObject(json, "provenance", provenance);

    char* json_string = cJSON_Print(json);
    cJSON_Delete(json);
    return json_string;
}

bool intent_frame_from_json(const char* json_str, IntentFrame* frame) {
    if (!json_str || !frame) {
        return false;
    }

    cJSON* json = cJSON_Parse(json_str);
    if (!json) {
        return false;
    }

    // Meta
    cJSON* meta = cJSON_GetObjectItem(json, "meta");
    if (meta) {
        cJSON* item = cJSON_GetObjectItem(meta, "ir_version");
        if (item) frame->meta.ir_version = (uint16_t)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(meta, "intent_id");
        if (item) frame->meta.intent_id = (uint64_t)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(meta, "session_id");
        if (item) frame->meta.session_id = (uint64_t)cJSON_GetNumberValue(item);
    }

    // Emotion
    cJSON* emotion = cJSON_GetObjectItem(json, "emotion");
    if (emotion) {
        cJSON* item = cJSON_GetObjectItem(emotion, "valence");
        if (item) frame->emotion.valence = (float)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(emotion, "arousal");
        if (item) frame->emotion.arousal = (float)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(emotion, "dominance");
        if (item) frame->emotion.dominance = (float)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(emotion, "discrete_id");
        if (item) frame->emotion.discrete_id = (int16_t)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(emotion, "intensity");
        if (item) frame->emotion.intensity = (float)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(emotion, "confidence");
        if (item) frame->emotion.confidence = (float)cJSON_GetNumberValue(item);
    }

    // Music
    cJSON* music = cJSON_GetObjectItem(json, "music");
    if (music) {
        cJSON* item = cJSON_GetObjectItem(music, "tempo_bias");
        if (item) frame->music.tempo_bias = (float)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(music, "rhythmic_density");
        if (item) frame->music.rhythmic_density = (float)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(music, "groove_strength");
        if (item) frame->music.groove_strength = (float)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(music, "harmonic_tension");
        if (item) frame->music.harmonic_tension = (float)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(music, "harmonic_motion");
        if (item) frame->music.harmonic_motion = (float)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(music, "mode_preference");
        if (item) frame->music.mode_preference = (int8_t)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(music, "melodic_activity");
        if (item) frame->music.melodic_activity = (float)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(music, "contour_variance");
        if (item) frame->music.contour_variance = (float)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(music, "dynamic_range");
        if (item) frame->music.dynamic_range = (float)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(music, "texture_density");
        if (item) frame->music.texture_density = (float)cJSON_GetNumberValue(item);
    }

    // Time
    cJSON* time = cJSON_GetObjectItem(json, "time");
    if (time) {
        cJSON* item = cJSON_GetObjectItem(time, "start_bar");
        if (item) frame->time.start_bar = (int32_t)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(time, "end_bar");
        if (item) frame->time.end_bar = (int32_t)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(time, "fade_in_beats");
        if (item) frame->time.fade_in_beats = (float)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(time, "fade_out_beats");
        if (item) frame->time.fade_out_beats = (float)cJSON_GetNumberValue(item);
    }

    // Constraints
    cJSON* constraints = cJSON_GetObjectItem(json, "constraints");
    if (constraints) {
        cJSON* item = cJSON_GetObjectItem(constraints, "allowed_engines_mask");
        if (item) frame->constraints.allowed_engines_mask = (uint32_t)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(constraints, "forbidden_engines_mask");
        if (item) frame->constraints.forbidden_engines_mask = (uint32_t)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(constraints, "max_cpu_cost");
        if (item) frame->constraints.max_cpu_cost = (float)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(constraints, "max_event_rate");
        if (item) frame->constraints.max_event_rate = (float)cJSON_GetNumberValue(item);
    }

    // Provenance
    cJSON* provenance = cJSON_GetObjectItem(json, "provenance");
    if (provenance) {
        cJSON* item = cJSON_GetObjectItem(provenance, "source");
        if (item) frame->provenance.source = (IntentSource)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(provenance, "user_override_weight");
        if (item) frame->provenance.user_override_weight = (float)cJSON_GetNumberValue(item);
    }

    cJSON_Delete(json);
    return true;
}

struct cJSON* intent_frame_to_cjson(const IntentFrame* frame) {
    if (!frame) {
        return nullptr;
    }

    cJSON* json = cJSON_CreateObject();
    if (!json) {
        return nullptr;
    }

    // Meta
    cJSON* meta = cJSON_CreateObject();
    cJSON_AddNumberToObject(meta, "ir_version", frame->meta.ir_version);
    cJSON_AddNumberToObject(meta, "intent_id", (double)frame->meta.intent_id);
    cJSON_AddNumberToObject(meta, "session_id", (double)frame->meta.session_id);
    cJSON_AddItemToObject(json, "meta", meta);

    // Emotion
    cJSON* emotion = cJSON_CreateObject();
    cJSON_AddNumberToObject(emotion, "valence", frame->emotion.valence);
    cJSON_AddNumberToObject(emotion, "arousal", frame->emotion.arousal);
    cJSON_AddNumberToObject(emotion, "dominance", frame->emotion.dominance);
    cJSON_AddNumberToObject(emotion, "discrete_id", frame->emotion.discrete_id);
    cJSON_AddNumberToObject(emotion, "intensity", frame->emotion.intensity);
    cJSON_AddNumberToObject(emotion, "confidence", frame->emotion.confidence);
    cJSON_AddItemToObject(json, "emotion", emotion);

    // Music
    cJSON* music = cJSON_CreateObject();
    cJSON_AddNumberToObject(music, "tempo_bias", frame->music.tempo_bias);
    cJSON_AddNumberToObject(music, "rhythmic_density", frame->music.rhythmic_density);
    cJSON_AddNumberToObject(music, "groove_strength", frame->music.groove_strength);
    cJSON_AddNumberToObject(music, "harmonic_tension", frame->music.harmonic_tension);
    cJSON_AddNumberToObject(music, "harmonic_motion", frame->music.harmonic_motion);
    cJSON_AddNumberToObject(music, "mode_preference", frame->music.mode_preference);
    cJSON_AddNumberToObject(music, "melodic_activity", frame->music.melodic_activity);
    cJSON_AddNumberToObject(music, "contour_variance", frame->music.contour_variance);
    cJSON_AddNumberToObject(music, "dynamic_range", frame->music.dynamic_range);
    cJSON_AddNumberToObject(music, "texture_density", frame->music.texture_density);
    cJSON_AddItemToObject(json, "music", music);

    // Time
    cJSON* time = cJSON_CreateObject();
    cJSON_AddNumberToObject(time, "start_bar", frame->time.start_bar);
    cJSON_AddNumberToObject(time, "end_bar", frame->time.end_bar);
    cJSON_AddNumberToObject(time, "fade_in_beats", frame->time.fade_in_beats);
    cJSON_AddNumberToObject(time, "fade_out_beats", frame->time.fade_out_beats);
    cJSON_AddItemToObject(json, "time", time);

    // Constraints
    cJSON* constraints = cJSON_CreateObject();
    cJSON_AddNumberToObject(constraints, "allowed_engines_mask", (double)frame->constraints.allowed_engines_mask);
    cJSON_AddNumberToObject(constraints, "forbidden_engines_mask", (double)frame->constraints.forbidden_engines_mask);
    cJSON_AddNumberToObject(constraints, "max_cpu_cost", frame->constraints.max_cpu_cost);
    cJSON_AddNumberToObject(constraints, "max_event_rate", frame->constraints.max_event_rate);
    cJSON_AddItemToObject(json, "constraints", constraints);

    // Provenance
    cJSON* provenance = cJSON_CreateObject();
    cJSON_AddNumberToObject(provenance, "source", frame->provenance.source);
    cJSON_AddNumberToObject(provenance, "user_override_weight", frame->provenance.user_override_weight);
    cJSON_AddItemToObject(json, "provenance", provenance);

    return json;
}

bool intent_frame_from_cjson(const struct cJSON* json, IntentFrame* frame) {
    if (!json || !frame) {
        return false;
    }

    // Meta
    cJSON* meta = cJSON_GetObjectItem(json, "meta");
    if (meta) {
        cJSON* item = cJSON_GetObjectItem(meta, "ir_version");
        if (item) frame->meta.ir_version = (uint16_t)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(meta, "intent_id");
        if (item) frame->meta.intent_id = (uint64_t)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(meta, "session_id");
        if (item) frame->meta.session_id = (uint64_t)cJSON_GetNumberValue(item);
    }

    // Emotion
    cJSON* emotion = cJSON_GetObjectItem(json, "emotion");
    if (emotion) {
        cJSON* item = cJSON_GetObjectItem(emotion, "valence");
        if (item) frame->emotion.valence = (float)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(emotion, "arousal");
        if (item) frame->emotion.arousal = (float)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(emotion, "dominance");
        if (item) frame->emotion.dominance = (float)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(emotion, "discrete_id");
        if (item) frame->emotion.discrete_id = (int16_t)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(emotion, "intensity");
        if (item) frame->emotion.intensity = (float)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(emotion, "confidence");
        if (item) frame->emotion.confidence = (float)cJSON_GetNumberValue(item);
    }

    // Music
    cJSON* music = cJSON_GetObjectItem(json, "music");
    if (music) {
        cJSON* item = cJSON_GetObjectItem(music, "tempo_bias");
        if (item) frame->music.tempo_bias = (float)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(music, "rhythmic_density");
        if (item) frame->music.rhythmic_density = (float)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(music, "groove_strength");
        if (item) frame->music.groove_strength = (float)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(music, "harmonic_tension");
        if (item) frame->music.harmonic_tension = (float)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(music, "harmonic_motion");
        if (item) frame->music.harmonic_motion = (float)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(music, "mode_preference");
        if (item) frame->music.mode_preference = (int8_t)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(music, "melodic_activity");
        if (item) frame->music.melodic_activity = (float)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(music, "contour_variance");
        if (item) frame->music.contour_variance = (float)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(music, "dynamic_range");
        if (item) frame->music.dynamic_range = (float)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(music, "texture_density");
        if (item) frame->music.texture_density = (float)cJSON_GetNumberValue(item);
    }

    // Time
    cJSON* time = cJSON_GetObjectItem(json, "time");
    if (time) {
        cJSON* item = cJSON_GetObjectItem(time, "start_bar");
        if (item) frame->time.start_bar = (int32_t)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(time, "end_bar");
        if (item) frame->time.end_bar = (int32_t)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(time, "fade_in_beats");
        if (item) frame->time.fade_in_beats = (float)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(time, "fade_out_beats");
        if (item) frame->time.fade_out_beats = (float)cJSON_GetNumberValue(item);
    }

    // Constraints
    cJSON* constraints = cJSON_GetObjectItem(json, "constraints");
    if (constraints) {
        cJSON* item = cJSON_GetObjectItem(constraints, "allowed_engines_mask");
        if (item) frame->constraints.allowed_engines_mask = (uint32_t)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(constraints, "forbidden_engines_mask");
        if (item) frame->constraints.forbidden_engines_mask = (uint32_t)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(constraints, "max_cpu_cost");
        if (item) frame->constraints.max_cpu_cost = (float)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(constraints, "max_event_rate");
        if (item) frame->constraints.max_event_rate = (float)cJSON_GetNumberValue(item);
    }

    // Provenance
    cJSON* provenance = cJSON_GetObjectItem(json, "provenance");
    if (provenance) {
        cJSON* item = cJSON_GetObjectItem(provenance, "source");
        if (item) frame->provenance.source = (IntentSource)cJSON_GetNumberValue(item);
        item = cJSON_GetObjectItem(provenance, "user_override_weight");
        if (item) frame->provenance.user_override_weight = (float)cJSON_GetNumberValue(item);
    }

    return true;
}

#else
// cJSON not available - stub implementations

char* intent_frame_to_json(const IntentFrame* frame) {
    (void)frame;  // Unused
    return nullptr;
}

bool intent_frame_from_json(const char* json_str, IntentFrame* frame) {
    (void)json_str;  // Unused
    (void)frame;     // Unused
    return false;
}

struct cJSON* intent_frame_to_cjson(const IntentFrame* frame) {
    (void)frame;  // Unused
    return nullptr;
}

bool intent_frame_from_cjson(const struct cJSON* json, IntentFrame* frame) {
    (void)json;   // Unused
    (void)frame;  // Unused
    return false;
}

#endif  // HAVE_CJSON

} // extern "C"
