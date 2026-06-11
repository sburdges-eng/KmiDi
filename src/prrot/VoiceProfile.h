#pragma once

/**
 * VoiceProfile.h - Parametric Voice Profile Data Structures
 * ==========================================================
 *
 * PRROT/PARROT Voice-Instrument Compiler - Tier C (Embedded Core)
 *
 * Defines data structures for voice profiles extracted from reference audio.
 * Voice profiles contain parametric characteristics that describe how a
 * specific voice produces sound at an abstract level, without storing
 * sentence-level waveforms or enabling waveform reconstruction.
 *
 * RT-Safe: These structures are designed for use in real-time audio contexts,
 * with fixed-size arrays and no dynamic allocation.
 */

#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

namespace prrot {

// =============================================================================
// CMU Phoneme Set (39 phonemes + silence)
// =============================================================================

constexpr size_t kMaxPhonemes = 40; // 39 CMU phonemes + SIL
constexpr size_t kMaxPhonemeSymbolLength = 4; // "SIL", "CH", etc.

// CMU Phoneme symbols
enum class PhonemeType : uint8_t {
    // Vowels
    AA = 0, AE, AH, AO, AW, AY, EH, ER, EY, IH, IY, OW, OY, UH, UW,
    // Consonants - Stops
    B, D, G, K, P, T,
    // Consonants - Fricatives
    CH, DH, F, HH, JH, S, SH, TH, V, Z, ZH,
    // Consonants - Nasals
    M, N, NG,
    // Consonants - Liquids/Glides
    L, R, W, Y,
    // Silence
    SIL,
    // Invalid/Unknown
    UNKNOWN = 255
};

// =============================================================================
// Phoneme Duration Statistics
// =============================================================================

struct PhonemeDurationStats {
    float mean_ms = 0.0f;      // Mean duration in milliseconds
    float variance_ms = 0.0f;  // Variance in milliseconds
    float min_ms = 0.0f;       // Minimum observed duration
    float max_ms = 0.0f;       // Maximum observed duration
    uint32_t sample_count = 0; // Number of samples used to compute stats
};

// =============================================================================
// Vowel Sustain Characteristics
// =============================================================================

struct VowelSustainCharacteristics {
    float sustain_multiplier = 1.0f;  // How much longer vowels are sustained relative to baseline
    float decay_rate = 0.0f;          // Energy decay rate during sustain (dB/ms)
    float formant_stability = 0.0f;   // How stable formants are during sustain [0-1]
};

// =============================================================================
// Consonant Attack/Release Profiles
// =============================================================================

struct ConsonantProfile {
    float attack_time_ms = 0.0f;    // Attack time for consonant onset
    float release_time_ms = 0.0f;   // Release time for consonant offset
    float strength_scalar = 1.0f;   // Relative strength of consonant articulation
    float burst_energy = 0.0f;      // Energy in burst/plosive release (for stops)
};

// =============================================================================
// Transition Statistics
// =============================================================================

struct TransitionStats {
    float mean_transition_time_ms = 0.0f;  // Mean transition duration
    float variance_ms = 0.0f;              // Variance in transition time
    float smoothness = 0.0f;                // Smoothness of transition [0-1]
    uint32_t sample_count = 0;
};

// =============================================================================
// Vibrato Characteristics
// =============================================================================

struct VibratoCharacteristics {
    float rate_mean_hz = 5.5f;        // Mean vibrato rate (Hz)
    float rate_variance_hz = 1.0f;    // Variance in vibrato rate
    float depth_mean_cents = 15.0f;   // Mean vibrato depth (cents)
    float depth_variance_cents = 5.0f;// Variance in vibrato depth
    float onset_time_ms = 50.0f;      // Typical time to reach full vibrato
    float probability = 0.7f;         // Probability of vibrato on sustained notes [0-1]
};

// =============================================================================
// Pitch Stability Profile
// =============================================================================

struct PitchStabilityProfile {
    float drift_rate_cents_per_sec = 0.0f;  // Average pitch drift rate
    float stability_variance = 0.0f;        // Variance in pitch stability [0-1]
    float correction_rate = 0.0f;           // How quickly pitch corrections occur [0-1]
};

// =============================================================================
// Articulation Variance
// =============================================================================

struct ArticulationVariance {
    float consistency_score = 0.0f;  // Overall articulation consistency [0-1]
    float variance_range_min = 0.0f; // Minimum variance range
    float variance_range_max = 0.0f; // Maximum variance range
    std::map<uint8_t, float> per_phoneme_variance; // Variance per phoneme type
};

// =============================================================================
// Prominence Tendencies
// =============================================================================

struct ProminenceCharacteristics {
    float stress_amplitude_multiplier = 1.0f;  // Amplitude increase for stressed syllables
    float stress_duration_multiplier = 1.2f;   // Duration increase for stressed syllables
    float emphasis_probability = 0.3f;         // Probability of applying emphasis
};

// =============================================================================
// Breath and Noise Characteristics
// =============================================================================

struct BreathNoiseCharacteristics {
    float breath_weight = 0.1f;      // Weight of breath noise in voice [0-1]
    float noise_floor_db = -40.0f;   // Typical noise floor level
    float breath_marker_probability = 0.2f; // Probability of breath markers at phrase boundaries
    float typical_breath_duration_ms = 200.0f; // Typical breath duration
};

// =============================================================================
// Phoneme Bias Table
// =============================================================================

// Bias table for phoneme production tendencies
// Positive values indicate preference/strength, negative indicate avoidance/weakness
struct PhonemeBiasTable {
    std::array<float, kMaxPhonemes> biases; // Bias values per phoneme [-1.0, 1.0]

    PhonemeBiasTable() {
        biases.fill(0.0f); // Neutral by default
    }

    float getBias(PhonemeType phoneme) const {
        if (static_cast<uint8_t>(phoneme) < kMaxPhonemes) {
            return biases[static_cast<uint8_t>(phoneme)];
        }
        return 0.0f;
    }

    void setBias(PhonemeType phoneme, float bias) {
        if (static_cast<uint8_t>(phoneme) < kMaxPhonemes) {
            biases[static_cast<uint8_t>(phoneme)] = std::clamp(bias, -1.0f, 1.0f);
        }
    }
};

// =============================================================================
// Voice Profile (Main Structure)
// =============================================================================

/**
 * VoiceProfile - Complete parametric voice profile
 *
 * Contains all extracted characteristics from reference audio.
 * Explicitly does NOT contain:
 * - Sentence-level waveforms
 * - Reusable audio chunks
 * - Any data that would enable waveform reconstruction of spoken content
 */
struct VoiceProfile {
    // Metadata
    std::string profile_id;                    // Unique identifier
    std::string profile_name;                  // Human-readable name
    uint64_t created_timestamp = 0;           // Unix timestamp of creation
    uint32_t version = 1;                     // Profile format version

    // Phoneme inventory (which phonemes this voice can produce)
    std::vector<PhonemeType> phoneme_inventory;

    // Phoneme duration distributions (per phoneme)
    std::map<PhonemeType, PhonemeDurationStats> phoneme_durations;

    // Vowel sustain characteristics
    VowelSustainCharacteristics vowel_sustain;

    // Consonant attack/release profiles (per consonant type)
    std::map<PhonemeType, ConsonantProfile> consonant_profiles;

    // Transition statistics
    std::map<std::pair<PhonemeType, PhonemeType>, TransitionStats> transitions; // (from, to) -> stats

    // Breath and noise weighting
    BreathNoiseCharacteristics breath_noise;

    // Vibrato tendencies
    VibratoCharacteristics vibrato;

    // Pitch stability vs. drift
    PitchStabilityProfile pitch_stability;

    // Articulation consistency vs. variance
    ArticulationVariance articulation_variance;

    // Prominence tendencies
    ProminenceCharacteristics prominence;

    // Phoneme bias table
    PhonemeBiasTable phoneme_bias;

    // Optional: Speech impediment characteristics (if detected/configured)
    // These are representable as articulation biases, never assumed by default
    std::map<std::string, float> articulation_biases; // e.g., "lisp_s", "rhotacism_r"

    // Validation
    bool isValid() const {
        return !profile_id.empty() && version > 0;
    }

    // Get phoneme duration stats (with defaults)
    PhonemeDurationStats getPhonemeDuration(PhonemeType phoneme) const {
        auto it = phoneme_durations.find(phoneme);
        if (it != phoneme_durations.end()) {
            return it->second;
        }
        return PhonemeDurationStats{}; // Default stats
    }

    // Get consonant profile (with defaults)
    ConsonantProfile getConsonantProfile(PhonemeType phoneme) const {
        auto it = consonant_profiles.find(phoneme);
        if (it != consonant_profiles.end()) {
            return it->second;
        }
        return ConsonantProfile{}; // Default profile
    }

    // Get transition stats (with defaults)
    TransitionStats getTransition(PhonemeType from, PhonemeType to) const {
        auto key = std::make_pair(from, to);
        auto it = transitions.find(key);
        if (it != transitions.end()) {
            return it->second;
        }
        return TransitionStats{}; // Default transition
    }
};

// =============================================================================
// Utility Functions
// =============================================================================

// Convert phoneme type to string symbol
const char* phonemeTypeToString(PhonemeType type);

// Convert string symbol to phoneme type
PhonemeType stringToPhonemeType(const std::string& symbol);

// Check if phoneme is a vowel
bool isVowel(PhonemeType phoneme);

// Check if phoneme is a consonant
bool isConsonant(PhonemeType phoneme);

} // namespace prrot
