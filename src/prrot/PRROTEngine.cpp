#include "prrot/PRROTEngine.h"
#include "prrot/VoiceProfile.h"
#include "prrot/PhonemeControlData.h"
#include "prrot/PitchTracker.h"
#include "prrot/AudioValidator.h"
#include "prrot/InputValidation.h"
#include "prrot/ProcessingError.h"
#include "penta/common/RTLogger.h"
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace prrot {

PRROTEngine::PRROTEngine() {
    // Create components
    phoneme_segmenter_ = std::make_unique<PhonemeSegmenter>();
    articulation_analyzer_ = std::make_unique<ArticulationAnalyzer>();
    envelope_generator_ = std::make_unique<EnvelopeGenerator>();
    spectral_analyzer_ = std::make_unique<SpectralAnalyzer>();
    breath_detector_ = std::make_unique<BreathDetector>();
    variance_modeler_ = std::make_unique<VarianceModeler>();
    midi_shaper_ = std::make_unique<MidiShaper>();
    pitch_tracker_ = std::make_unique<PitchTracker>();
    audio_validator_ = std::make_unique<AudioValidator>();
}

PRROTEngine::~PRROTEngine() = default;

bool PRROTEngine::initialize() {
    // Initialize components
    // Note: Memory pools would be initialized here in production
    // For now, components use stack-allocated buffers

    return true;
}

bool PRROTEngine::loadVoiceProfile(const VoiceProfile& profile) {
    if (!profile.isValid()) {
        return false;
    }

    voice_profile_ = profile;
    profile_loaded_ = true;

    return true;
}

PhonemeControlData PRROTEngine::processAudioSegment(
    const float* audio_samples,
    size_t num_samples,
    float sample_rate_hz,
    float tempo_bpm
) noexcept {
    PhonemeControlData control_data;
    control_data.tempo_bpm = tempo_bpm;
    control_data.sample_rate_hz = sample_rate_hz;

    // Comprehensive input validation. RT-path: log with static context only.
    auto input_validation = validateAudioInput(audio_samples, num_samples, sample_rate_hz);
    if (input_validation.hasErrors()) {
        penta::getLogger().logRT(penta::LogLevel::Error,
            "PRROTEngine::processAudioSegment: Input validation failed");
        return control_data;
    }

    if (input_validation.hasWarnings()) {
        penta::getLogger().logRT(penta::LogLevel::Warning,
            "PRROTEngine::processAudioSegment: Input validation produced warnings");
    }

    // Validate audio quality
    auto validation = audio_validator_->validate(audio_samples, num_samples, sample_rate_hz);
    if (!validation.is_valid) {
        penta::getLogger().logRT(penta::LogLevel::Error,
            "PRROTEngine::processAudioSegment: Audio validation failed");
        return control_data;
    }

    if (validation.quality_score() < 0.5f) {
        char msg[128];
        std::snprintf(msg, sizeof(msg),
            "PRROTEngine::processAudioSegment: Low audio quality (score: %.3f)",
            validation.quality_score());
        penta::getLogger().logRT(penta::LogLevel::Warning, msg);
    }

    // Analyze phonemes with error recovery
    auto phoneme_sequence = analyzePhonemes(audio_samples, num_samples, sample_rate_hz);
    control_data.phoneme_sequence = phoneme_sequence;

    // Log if phoneme analysis failed
    if (phoneme_sequence.empty()) {
        penta::getLogger().logRT(penta::LogLevel::Warning,
            "PRROTEngine::processAudioSegment: No phonemes detected");
    }

    // Detect breath markers
    control_data.breath_markers = detectBreathMarkers(audio_samples, num_samples, sample_rate_hz);

    // Track pitch using real pitch tracker (replaces placeholder)
    if (!phoneme_sequence.empty()) {
        // Track pitch for each phoneme segment
        for (const auto& phoneme : phoneme_sequence) {
            size_t start_sample = static_cast<size_t>((phoneme.start_time_ms / 1000.0f) * sample_rate_hz);
            size_t duration_samples = static_cast<size_t>((phoneme.duration_ms / 1000.0f) * sample_rate_hz);
            size_t end_sample = std::min(start_sample + duration_samples, num_samples);

            if (start_sample < num_samples && duration_samples > 0) {
                auto pitch_result = pitch_tracker_->trackPitch(
                    audio_samples + start_sample,
                    end_sample - start_sample,
                    sample_rate_hz
                );

                if (pitch_result.is_valid && pitch_result.confidence > 0.3f) {
                    PitchTarget target;
                    target.time_ms = phoneme.start_time_ms;
                    target.midi_note = pitch_result.midi_note;
                    target.cents_offset = pitch_result.cents_offset;
                    target.confidence = pitch_result.confidence;
                    control_data.pitch_targets.push_back(target);
                } else {
                    // Fallback: Use phoneme-based pitch estimation
                    if (pitch_result.is_valid && pitch_result.confidence <= 0.3f) {
                        char msg[128];
                        std::snprintf(msg, sizeof(msg),
                            "PRROTEngine: Low pitch confidence (%.3f), using fallback",
                            pitch_result.confidence);
                        penta::getLogger().logRT(penta::LogLevel::Debug, msg);
                    }
                    PitchTarget target;
                    target.time_ms = phoneme.start_time_ms;
                    target.midi_note = estimatePitchFromPhoneme(phoneme.phoneme);
                    target.cents_offset = 0.0f;
                    target.confidence = 0.5f; // Lower confidence for fallback
                    control_data.pitch_targets.push_back(target);
                }
            }
        }
    } else {
        // If no phonemes detected, try pitch tracking on entire segment
        auto pitch_result = pitch_tracker_->trackPitch(audio_samples, num_samples, sample_rate_hz);
        if (pitch_result.is_valid) {
            PitchTarget target;
            target.time_ms = 0.0f;
            target.midi_note = pitch_result.midi_note;
            target.cents_offset = pitch_result.cents_offset;
            target.confidence = pitch_result.confidence;
            control_data.pitch_targets.push_back(target);
        }
    }

    // Generate MIDI notes
    if (profile_loaded_) {
        control_data.midi_notes = midi_shaper_->shapeMidiNotes(
            phoneme_sequence,
            control_data.pitch_targets,
            voice_profile_,
            tempo_bpm
        );
    }

    // Generate articulation envelopes
    generateArticulationEnvelopes(control_data, phoneme_sequence);

    // Generate automation envelopes
    generateAutomationEnvelopes(control_data, phoneme_sequence);

    return control_data;
}

PhonemeControlData PRROTEngine::generateControlData(
    const std::vector<PhonemeTiming>& phoneme_sequence,
    const std::vector<PitchTarget>& pitch_targets,
    float tempo_bpm
) noexcept {
    PhonemeControlData control_data;
    control_data.tempo_bpm = tempo_bpm;
    control_data.phoneme_sequence = phoneme_sequence;
    control_data.pitch_targets = pitch_targets;

    if (!profile_loaded_ || phoneme_sequence.empty()) {
        return control_data;
    }

    // Generate MIDI notes
    control_data.midi_notes = midi_shaper_->shapeMidiNotes(
        phoneme_sequence,
        pitch_targets,
        voice_profile_,
        tempo_bpm
    );

    // Generate articulation envelopes
    generateArticulationEnvelopes(control_data, phoneme_sequence);

    // Generate automation envelopes
    generateAutomationEnvelopes(control_data, phoneme_sequence);

    return control_data;
}

std::vector<PhonemeTiming> PRROTEngine::analyzePhonemes(
    const float* audio_samples,
    size_t num_samples,
    float sample_rate_hz
) noexcept {
    std::vector<PhonemeTiming> phoneme_timings;

    // Validate inputs. RT-path: static log context.
    auto validation = validateAudioInput(audio_samples, num_samples, sample_rate_hz);
    if (validation.hasErrors()) {
        penta::getLogger().logRT(penta::LogLevel::Error,
            "PRROTEngine::analyzePhonemes: Input validation failed");
        return phoneme_timings;
    }

    auto segment_result = phoneme_segmenter_->segment(audio_samples, num_samples, sample_rate_hz);

    if (!segment_result.valid) {
        penta::getLogger().logRT(penta::LogLevel::Warning,
            "PRROTEngine::analyzePhonemes: Segmentation failed");
        return phoneme_timings;
    }

    if (segment_result.confidence < 0.5f) {
        char msg[128];
        std::snprintf(msg, sizeof(msg),
            "PRROTEngine::analyzePhonemes: Low segmentation confidence (%.3f)",
            segment_result.confidence);
        penta::getLogger().logRT(penta::LogLevel::Debug, msg);
    }

    // Convert segments to phoneme timings
    for (size_t i = 0; i < segment_result.phonemes.size() && i + 1 < segment_result.boundaries_ms.size(); ++i) {
        PhonemeTiming timing;
        timing.phoneme = segment_result.phonemes[i];
        timing.start_time_ms = segment_result.boundaries_ms[i];
        timing.duration_ms = segment_result.boundaries_ms[i + 1] - segment_result.boundaries_ms[i];

        // Detect onset/offset
        size_t start_sample = static_cast<size_t>((timing.start_time_ms / 1000.0f) * sample_rate_hz);
        size_t duration_samples = static_cast<size_t>((timing.duration_ms / 1000.0f) * sample_rate_hz);

        if (start_sample < num_samples && duration_samples > 0) {
            size_t end_sample = std::min(start_sample + duration_samples, num_samples);
            auto onset_info = articulation_analyzer_->detectOnset(
                audio_samples + start_sample,
                end_sample - start_sample,
                sample_rate_hz
            );
            auto offset_info = articulation_analyzer_->detectOffset(
                audio_samples + start_sample,
                end_sample - start_sample,
                sample_rate_hz
            );

            timing.onset_time_ms = onset_info.time_ms;
            timing.offset_time_ms = offset_info.time_ms;
        }

        phoneme_timings.push_back(timing);
    }

    return phoneme_timings;
}

std::vector<BreathMarker> PRROTEngine::detectBreathMarkers(
    const float* audio_samples,
    size_t num_samples,
    float sample_rate_hz
) noexcept {
    std::vector<BreathMarker> markers;

    // Validate inputs. RT-path: static log context.
    auto validation = validateAudioInput(audio_samples, num_samples, sample_rate_hz);
    if (validation.hasErrors()) {
        penta::getLogger().logRT(penta::LogLevel::Error,
            "PRROTEngine::detectBreathMarkers: Input validation failed");
        return markers;
    }

    // Detect breath in segments
    size_t segment_size = static_cast<size_t>(sample_rate_hz * 1.0f); // 1 second segments

    for (size_t offset = 0; offset < num_samples; offset += segment_size) {
        size_t segment_samples = std::min(segment_size, num_samples - offset);

        auto marker = breath_detector_->detectBreath(
            audio_samples + offset,
            segment_samples,
            sample_rate_hz
        );

        if (marker.confidence > 0.5f) {
            // Adjust time to account for segment offset
            marker.time_ms += (static_cast<float>(offset) / sample_rate_hz) * 1000.0f;
            // Convert BreathDetector::BreathMarker to PhonemeControlData::BreathMarker
            BreathMarker bm;
            bm.time_ms = marker.time_ms;
            bm.intensity = marker.intensity;
            bm.duration_ms = marker.duration_ms;
            markers.push_back(bm);
        }
    }

    return markers;
}

void PRROTEngine::generateArticulationEnvelopes(
    PhonemeControlData& control_data,
    const std::vector<PhonemeTiming>& phoneme_sequence
) noexcept {
    if (!profile_loaded_) {
        return;
    }

    control_data.articulation_envelopes.clear();
    control_data.articulation_envelopes.reserve(phoneme_sequence.size());

    for (const auto& phoneme_timing : phoneme_sequence) {
        ArticulationEnvelope envelope;
        envelope.phoneme = phoneme_timing.phoneme;

        // Get articulation shape from analyzer
        // Note: Would need audio samples for full analysis
        // For now, use defaults based on phoneme type

        if (prrot::isVowel(phoneme_timing.phoneme)) {
            envelope.attack_time_ms = 20.0f;
            envelope.sustain_level = 1.0f;
            envelope.release_time_ms = 50.0f;
            envelope.vowel_sustain_mult = voice_profile_.vowel_sustain.sustain_multiplier;
        } else if (prrot::isConsonant(phoneme_timing.phoneme)) {
            auto consonant_profile = voice_profile_.getConsonantProfile(phoneme_timing.phoneme);
            envelope.attack_time_ms = consonant_profile.attack_time_ms;
            envelope.release_time_ms = consonant_profile.release_time_ms;
            envelope.consonant_strength = consonant_profile.strength_scalar;
        }

        control_data.articulation_envelopes.push_back(envelope);
    }
}

void PRROTEngine::generateAutomationEnvelopes(
    PhonemeControlData& control_data,
    const std::vector<PhonemeTiming>& phoneme_sequence
) noexcept {
    if (!profile_loaded_ || phoneme_sequence.empty()) {
        return;
    }

    // Generate dynamics envelope
    AutomationEnvelope dynamics;
    dynamics.type = AutomationType::Dynamics;
    dynamics.interpolation = AutomationEnvelope::Interpolation::Smooth;

    float total_duration_ms = 0.0f;
    for (const auto& phoneme : phoneme_sequence) {
        total_duration_ms = std::max(total_duration_ms, phoneme.start_time_ms + phoneme.duration_ms);
    }

    // Add points based on phoneme prominence
    for (const auto& phoneme : phoneme_sequence) {
        float value = phoneme.is_stressed ? 1.0f : 0.7f;
        value *= phoneme.prominence;
        dynamics.addPoint(phoneme.start_time_ms, value);
        dynamics.addPoint(phoneme.start_time_ms + phoneme.duration_ms, value * 0.9f);
    }

    control_data.automation_envelopes.push_back(dynamics);

    // Generate expression envelope (similar to dynamics)
    AutomationEnvelope expression;
    expression.type = AutomationType::Expression;
    expression.interpolation = AutomationEnvelope::Interpolation::Linear;

    for (const auto& phoneme : phoneme_sequence) {
        float value = phoneme.is_emphasized ? 1.0f : 0.8f;
        expression.addPoint(phoneme.start_time_ms, value);
    }

    control_data.automation_envelopes.push_back(expression);
}

int PRROTEngine::estimatePitchFromPhoneme(PhonemeType phoneme) const noexcept {
    // Fallback pitch estimation based on phoneme type
    // Vowels typically have lower pitch, consonants higher
    // This is a simplified heuristic - real pitch tracking is preferred

    if (prrot::isVowel(phoneme)) {
        // Vowels: typically in lower-mid range (MIDI 60-72, C4-C5)
        return 66; // A4# (typical vowel pitch)
    } else if (prrot::isConsonant(phoneme)) {
        // Consonants: can vary, use middle C as default
        return 60; // C4
    }

    // Unknown: default to middle C
    return 60;
}

} // namespace prrot
