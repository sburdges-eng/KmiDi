#pragma once

/**
 * PRROTEngine.h - Main PRROT/PARROT Engine API (Public Header)
 * ============================================================
 *
 * Public API header for PRROT/PARROT voice-instrument compiler.
 *
 * RT-Safe: All methods use pre-allocated buffers and are safe to call
 * from audio callbacks (though the engine should be initialized before
 * audio processing starts).
 */

#include "prrot/VoiceProfile.h"
#include "prrot/PhonemeControlData.h"
#include <memory>
#include <vector>

// Forward declarations
namespace prrot {
    class PhonemeSegmenter;
    class ArticulationAnalyzer;
    class EnvelopeGenerator;
    class SpectralAnalyzer;
    class BreathDetector;
    class VarianceModeler;
    class MidiShaper;
}

namespace prrot {

/**
 * PRROTEngine - Main embedded engine
 *
 * Provides RT-safe voice analysis and control data generation.
 * All components use pre-allocated buffers - no dynamic allocation
 * during processing.
 */
class PRROTEngine {
public:
    PRROTEngine();
    ~PRROTEngine();

    // Initialize engine (must be called before use, typically at startup)
    bool initialize();

    // Load voice profile (non-RT operation, should be done before audio processing)
    bool loadVoiceProfile(const VoiceProfile& profile);

    // Get current voice profile
    const VoiceProfile& getVoiceProfile() const { return voice_profile_; }

    // Process audio segment to generate control data
    PhonemeControlData processAudioSegment(
        const float* audio_samples,
        size_t num_samples,
        float sample_rate_hz,
        float tempo_bpm = 120.0f
    );

    // Generate control data from phoneme sequence and pitch targets
    // RT-Safe: Uses pre-allocated buffers only
    PhonemeControlData generateControlData(
        const std::vector<PhonemeTiming>& phoneme_sequence,
        const std::vector<PitchTarget>& pitch_targets,
        float tempo_bpm = 120.0f
    ) noexcept;

    // Analyze audio segment and extract phoneme information
    // Returns std::vector (heap allocation); not noexcept.
    std::vector<PhonemeTiming> analyzePhonemes(
        const float* audio_samples,
        size_t num_samples,
        float sample_rate_hz
    );

    // Detect breath markers in audio
    // Returns std::vector (heap allocation); not noexcept.
    std::vector<BreathMarker> detectBreathMarkers(
        const float* audio_samples,
        size_t num_samples,
        float sample_rate_hz
    );

private:
    // Voice profile
    VoiceProfile voice_profile_;
    bool profile_loaded_ = false;

    // Tier C components
    std::unique_ptr<PhonemeSegmenter> phoneme_segmenter_;
    std::unique_ptr<ArticulationAnalyzer> articulation_analyzer_;
    std::unique_ptr<EnvelopeGenerator> envelope_generator_;
    std::unique_ptr<SpectralAnalyzer> spectral_analyzer_;
    std::unique_ptr<BreathDetector> breath_detector_;
    std::unique_ptr<VarianceModeler> variance_modeler_;
    std::unique_ptr<MidiShaper> midi_shaper_;

    // Internal helpers
    void generateArticulationEnvelopes(
        PhonemeControlData& control_data,
        const std::vector<PhonemeTiming>& phoneme_sequence
    ) noexcept;

    void generateAutomationEnvelopes(
        PhonemeControlData& control_data,
        const std::vector<PhonemeTiming>& phoneme_sequence
    ) noexcept;
};

} // namespace prrot
