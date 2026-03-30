#pragma once

/**
 * Kelly FFI Interface
 * ===================
 * C-compatible API for exposing KellyBrain functionality to other languages (Rust, etc.)
 * 
 * This provides a Foreign Function Interface (FFI) layer around the C++ KellyBrain
 * implementation, allowing integration with Tauri/Rust and other systems.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

// =============================================================================
// Opaque Types
// =============================================================================

/**
 * Opaque handle to KellyBrain instance
 * The actual C++ object is hidden behind this pointer
 */
typedef struct KellyBrain KellyBrain;

/**
 * Opaque handle to EmotionThesaurus instance
 */
typedef struct EmotionThesaurus EmotionThesaurus;

// =============================================================================
// Error Handling
// =============================================================================

/**
 * Error codes returned by FFI functions
 */
typedef enum {
    KELLY_SUCCESS = 0,
    KELLY_ERROR_NULL_POINTER = -1,
    KELLY_ERROR_INITIALIZATION_FAILED = -2,
    KELLY_ERROR_INVALID_PARAMETER = -3,
    KELLY_ERROR_JSON_PARSE_ERROR = -4,
    KELLY_ERROR_MEMORY_ALLOCATION = -5,
    KELLY_ERROR_FILE_NOT_FOUND = -6,
    KELLY_ERROR_UNKNOWN = -999
} KellyErrorCode;

/**
 * Get human-readable error message for error code
 * @param error_code Error code from KellyErrorCode enum
 * @return Static string describing the error (do not free)
 */
const char* kelly_get_error_message(KellyErrorCode error_code);

// =============================================================================
// Memory Management
// =============================================================================

/**
 * Free memory allocated by kelly_* functions
 * Use this to free any char* returned by kelly functions
 * @param ptr Pointer to free (can be NULL)
 */
void kelly_free_string(char* ptr);

// =============================================================================
// KellyBrain Core Functions
// =============================================================================

/**
 * Create new KellyBrain instance
 * @return Pointer to KellyBrain instance, or NULL on failure
 */
KellyBrain* kelly_brain_create(void);

/**
 * Initialize KellyBrain with data directory
 * @param brain KellyBrain instance
 * @param data_path Path to data directory containing emotion thesaurus data
 * @return KELLY_SUCCESS on success, error code on failure
 */
KellyErrorCode kelly_brain_initialize(KellyBrain* brain, const char* data_path);

/**
 * Check if KellyBrain is initialized and ready
 * @param brain KellyBrain instance
 * @return true if initialized, false otherwise
 */
bool kelly_brain_is_initialized(const KellyBrain* brain);

/**
 * Destroy KellyBrain instance and free memory
 * @param brain KellyBrain instance (can be NULL)
 */
void kelly_brain_destroy(KellyBrain* brain);

// =============================================================================
// Intent Processing Functions
// =============================================================================

/**
 * Process text description and generate intent result
 * @param brain KellyBrain instance
 * @param text Text description of emotional state/wound
 * @return JSON string containing IntentResult, or NULL on error (caller must call kelly_free_string)
 */
char* kelly_brain_from_text(KellyBrain* brain, const char* text);

/**
 * Process emotion name and generate intent result
 * @param brain KellyBrain instance  
 * @param emotion_name Name of emotion to process
 * @param intensity Intensity value (0.0 to 1.0)
 * @return JSON string containing IntentResult, or NULL on error (caller must call kelly_free_string)
 */
char* kelly_brain_from_emotion(KellyBrain* brain, const char* emotion_name, float intensity);

/**
 * Process journey from current state to desired state
 * @param brain KellyBrain instance
 * @param current_state_json JSON describing current emotional state (SideA)
 * @param desired_state_json JSON describing desired emotional state (SideB)
 * @return JSON string containing IntentResult, or NULL on error (caller must call kelly_free_string)
 */
char* kelly_brain_from_journey(KellyBrain* brain, const char* current_state_json, const char* desired_state_json);

/**
 * Process wound structure directly
 * @param brain KellyBrain instance
 * @param wound_json JSON string containing Wound structure
 * @return JSON string containing IntentResult, or NULL on error (caller must call kelly_free_string)
 */
char* kelly_brain_from_wound(KellyBrain* brain, const char* wound_json);

// =============================================================================
// MIDI Generation Functions
// =============================================================================

/**
 * Generate MIDI from intent result
 * @param brain KellyBrain instance
 * @param intent_json JSON string containing IntentResult
 * @param bars Number of bars to generate (default: 8)
 * @return JSON string containing GeneratedMidi, or NULL on error (caller must call kelly_free_string)
 */
char* kelly_brain_generate_midi(KellyBrain* brain, const char* intent_json, int bars);

/**
 * Generate MIDI with specific parameters
 * @param brain KellyBrain instance
 * @param intent_json JSON string containing IntentResult
 * @param bars Number of bars to generate
 * @param bpm Beats per minute
 * @param key_signature Key signature (e.g., "C", "Dm", "F#")
 * @return JSON string containing GeneratedMidi, or NULL on error (caller must call kelly_free_string)
 */
char* kelly_brain_generate_midi_with_params(KellyBrain* brain, const char* intent_json, 
                                          int bars, int bpm, const char* key_signature);

// =============================================================================
// State Query Functions
// =============================================================================

/**
 * Get current emotional state from KellyBrain
 * @param brain KellyBrain instance
 * @return JSON string containing current emotion state, or NULL on error (caller must call kelly_free_string)
 */
char* kelly_brain_get_emotion_state(const KellyBrain* brain);

/**
 * Get current processing parameters
 * @param brain KellyBrain instance
 * @return JSON string containing current parameters, or NULL on error (caller must call kelly_free_string)
 */
char* kelly_brain_get_parameters(const KellyBrain* brain);

/**
 * Get available emotions from thesaurus
 * @param brain KellyBrain instance
 * @return JSON string containing emotion list, or NULL on error (caller must call kelly_free_string)
 */
char* kelly_brain_get_available_emotions(const KellyBrain* brain);

// =============================================================================
// Parameter Update Functions
// =============================================================================

/**
 * Update emotional parameters
 * @param brain KellyBrain instance
 * @param valence Valence value (-1.0 to 1.0)
 * @param arousal Arousal value (0.0 to 1.0)
 * @param dominance Dominance value (0.0 to 1.0)
 * @return KELLY_SUCCESS on success, error code on failure
 */
KellyErrorCode kelly_brain_set_emotion_parameters(KellyBrain* brain, float valence, float arousal, float dominance);

/**
 * Update processing parameters from JSON
 * @param brain KellyBrain instance
 * @param params_json JSON string containing parameter updates
 * @return KELLY_SUCCESS on success, error code on failure
 */
KellyErrorCode kelly_brain_update_parameters(KellyBrain* brain, const char* params_json);

// =============================================================================
// Event/Callback System (for real-time updates)
// =============================================================================

/**
 * Callback function type for state change notifications
 * @param event_type Type of event (emotion_changed, midi_generated, etc.)
 * @param data JSON data associated with the event
 * @param user_data User-provided data pointer
 */
typedef void (*KellyEventCallback)(const char* event_type, const char* data, void* user_data);

/**
 * Register callback for state change events
 * @param brain KellyBrain instance
 * @param callback Callback function to call on events
 * @param user_data User data to pass to callback
 * @return KELLY_SUCCESS on success, error code on failure
 */
KellyErrorCode kelly_brain_register_callback(KellyBrain* brain, KellyEventCallback callback, void* user_data);

/**
 * Unregister event callback
 * @param brain KellyBrain instance
 * @return KELLY_SUCCESS on success, error code on failure
 */
KellyErrorCode kelly_brain_unregister_callback(KellyBrain* brain);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Get KellyBrain version information
 * @return Static string containing version info (do not free)
 */
const char* kelly_get_version(void);

/**
 * Check if required data files are available
 * @param data_path Path to data directory
 * @return true if all required files are present, false otherwise
 */
bool kelly_check_data_files(const char* data_path);

/**
 * Get last error message (thread-local)
 * @return String describing the last error, or NULL if no error (do not free)
 */
const char* kelly_get_last_error(void);

#ifdef __cplusplus
}
#endif