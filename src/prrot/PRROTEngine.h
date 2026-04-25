#pragma once

/**
 * PRROTEngine.h - Main PRROT/PARROT Engine API
 * ============================================
 *
 * PRROT/PARROT Voice-Instrument Compiler - Tier C (Embedded Core)
 *
 * Main API for PRROT/PARROT voice-instrument compiler.
 * Integrates all Tier C components for RT-safe voice analysis and
 * control data generation.
 *
 * RT-Safe: All methods use pre-allocated buffers and are safe to call
 * from audio callbacks (though the engine should be initialized before
 * audio processing starts).
 */

#include "prrot/VoiceProfile.h"
#include "prrot/PhonemeControlData.h"
#include "prrot/PhonemeSegmenter.h"
#include "prrot/ArticulationAnalyzer.h"
#include "prrot/EnvelopeGenerator.h"
#include "prrot/SpectralAnalyzer.h"
#include "prrot/BreathDetector.h"
#include "prrot/VarianceModeler.h"
#include "prrot/MidiShaper.h"
#include "prrot/PitchTracker.h"
#include "prrot/AudioValidator.h"
#include <memory>
#include <vector>

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
    // RT-Safe: Uses pre-allocated buffers only
    std::vector<PhonemeTiming> analyzePhonemes(
        const float* audio_samples,
        size_t num_samples,
        float sample_rate_hz
    );

    // Detect breath markers in audio
    // RT-Safe: Uses pre-allocated buffers only
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
    std::unique_ptr<PitchTracker> pitch_tracker_;
    std::unique_ptr<AudioValidator> audio_validator_;

    // Internal helpers
    void generateArticulationEnvelopes(
        PhonemeControlData& control_data,
        const std::vector<PhonemeTiming>& phoneme_sequence
    ) noexcept;

    void generateAutomationEnvelopes(
        PhonemeControlData& control_data,
        const std::vector<PhonemeTiming>& phoneme_sequence
    ) noexcept;

    /**
     * Estimate pitch from phoneme type (fallback method)
     */
    int estimatePitchFromPhoneme(PhonemeType phoneme) const noexcept;
};

} // namespace prrot
