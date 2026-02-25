#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <array>
#include <vector>
#include <memory>
#include <cmath>

/**
 * Wavetable Voice Synthesizer
 *
 * Inspired by Vaporizer2 and Surge XT wavetable architectures.
 * Combines wavetable synthesis with formant filtering for vocal sounds.
 *
 * Features:
 * - 256-sample wavetables with up to 256 frames
 * - Morphing between wavetable frames
 * - Formant filter bank overlay
 * - Unison with detune
 * - Modulation matrix
 */

//==============================================================================
// Constants
//==============================================================================
constexpr int WAVETABLE_SIZE = 256;        // Samples per wave
constexpr int MAX_WAVETABLE_FRAMES = 256;  // Max frames in wavetable
constexpr int MAX_UNISON_VOICES = 16;
constexpr int NUM_FORMANT_FILTERS = 5;

//==============================================================================
/**
 * Single wavetable containing multiple frames
 */
class Wavetable
{
public:
    Wavetable();

    /** Generate basic waveforms */
    void generateSine();
    void generateSaw();
    void generateSquare();
    void generateTriangle();
    void generatePulse(float width);

    /** Generate vocal formant wavetable */
    void generateVowelFormants(const std::array<float, 5>& formantFreqs);

    /** Generate morphing vowel wavetable (A-E-I-O-U across frames) */
    void generateVowelMorph();

    /** Load wavetable from data */
    void loadFromData(const float* data, int numFrames, int samplesPerFrame);

    /** Get interpolated sample */
    float getSample(float phase, float framePosition) const;

    /** Get number of frames */
    int getNumFrames() const { return numFrames_; }

private:
    std::vector<std::array<float, WAVETABLE_SIZE>> frames_;
    int numFrames_ = 1;

    // Interpolation helpers
    float interpolateFrame(const std::array<float, WAVETABLE_SIZE>& frame, float phase) const;
};

//==============================================================================
/**
 * Formant filter for shaping wavetable output
 */
class FormantFilterBank
{
public:
    FormantFilterBank();

    void setSampleRate(double sampleRate);

    /** Set formant frequencies (F1-F5) and bandwidths */
    void setFormants(const std::array<float, 5>& frequencies,
                     const std::array<float, 5>& bandwidths);

    /** Set vowel preset */
    void setVowel(int vowelIndex);  // 0=A, 1=E, 2=I, 3=O, 4=U

    /** Morph between two vowels */
    void morphVowels(int vowel1, int vowel2, float position);

    /** Process sample through formant filters */
    float process(float input);

    /** Reset filter states */
    void reset();

private:
    struct BiquadFilter
    {
        float a1 = 0, a2 = 0;
        float b0 = 0, b1 = 0, b2 = 0;
        float z1 = 0, z2 = 0;

        void setResonator(float frequency, float bandwidth, float sampleRate);
        float process(float input);
        void reset() { z1 = z2 = 0; }
    };

    std::array<BiquadFilter, NUM_FORMANT_FILTERS> filters_;
    std::array<float, NUM_FORMANT_FILTERS> gains_;
    double sampleRate_ = 44100.0;

    // Standard vowel formants (F1-F5 in Hz)
    static constexpr std::array<std::array<float, 5>, 5> VOWEL_FORMANTS = {{
        {{ 730, 1090, 2440, 3400, 4500 }},  // A
        {{ 570, 1980, 2440, 3400, 4500 }},  // E
        {{ 270, 2290, 3010, 3400, 4500 }},  // I
        {{ 570,  840, 2410, 3400, 4500 }},  // O
        {{ 300,  870, 2240, 3400, 4500 }}   // U
    }};
};

//==============================================================================
/**
 * Single wavetable oscillator voice with unison
 */
class WavetableVoice
{
public:
    WavetableVoice();

    void setSampleRate(double sampleRate);
    void setWavetable(std::shared_ptr<Wavetable> wavetable);

    /** Set oscillator parameters */
    void setFrequency(float frequency);
    void setWavetablePosition(float position);  // 0.0 to 1.0
    void setUnison(int numVoices, float detune);

    /** Envelope control */
    void noteOn(float velocity);
    void noteOff();

    /** Process single sample */
    float process();

    bool isActive() const { return active_; }

private:
    double sampleRate_ = 44100.0;
    std::shared_ptr<Wavetable> wavetable_;

    // Oscillator state
    float frequency_ = 440.0f;
    float wavetablePos_ = 0.0f;

    // Unison
    int unisonVoices_ = 1;
    float unisonDetune_ = 0.0f;
    std::array<float, MAX_UNISON_VOICES> phases_;
    std::array<float, MAX_UNISON_VOICES> detuneFactors_;

    // Envelope
    bool active_ = false;
    float envelope_ = 0.0f;
    float envelopeTarget_ = 0.0f;
    float attackRate_ = 0.01f;
    float releaseRate_ = 0.001f;

    void updateDetuneFactors();
};

//==============================================================================
/**
 * Complete wavetable voice synthesizer
 *
 * Combines:
 * - Wavetable oscillator with morphing
 * - Formant filter bank
 * - Modulation (vibrato, tremolo)
 * - Effects (chorus, saturation)
 */
class WavetableVoiceSynth : public juce::AudioProcessor
{
public:
    WavetableVoiceSynth();
    ~WavetableVoiceSynth() override = default;

    //==========================================================================
    // AudioProcessor overrides
    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    const juce::String getName() const override { return "DAiWWavetableVoice"; }
    bool hasEditor() const override { return false; }
    juce::AudioProcessorEditor* createEditor() override { return nullptr; }
    double getTailLengthSeconds() const override { return 0.5; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}

    void getStateInformation(juce::MemoryBlock&) override {}
    void setStateInformation(const void*, int) override {}

    //==========================================================================
    // Synthesis parameters (atomic for thread safety)

    /** Wavetable position (0.0 to 1.0) */
    std::atomic<float> wavetablePosition{ 0.0f };

    /** Formant vowel (0=A, 1=E, 2=I, 3=O, 4=U, fractional for morph) */
    std::atomic<float> vowelPosition{ 0.0f };

    /** Formant filter mix (0.0 = bypass, 1.0 = full) */
    std::atomic<float> formantMix{ 0.8f };

    /** Unison voices (1-16) */
    std::atomic<int> unisonVoices{ 4 };

    /** Unison detune in cents */
    std::atomic<float> unisonDetune{ 20.0f };

    /** Vibrato rate (Hz) */
    std::atomic<float> vibratoRate{ 5.5f };

    /** Vibrato depth (cents) */
    std::atomic<float> vibratoDepth{ 30.0f };

    /** Output gain (dB) */
    std::atomic<float> outputGain{ 0.0f };

    //==========================================================================
    // Voice control

    /** Trigger note on */
    void noteOn(int midiNote, float velocity);

    /** Trigger note off */
    void noteOff();

    /** Set current vowel (real-time) */
    void setVowel(int vowelIndex);

    /** Load custom wavetable */
    void loadWavetable(const float* data, int numFrames, int samplesPerFrame);

    /** Load preset wavetable */
    void loadPresetWavetable(const juce::String& presetName);

private:
    double sampleRate_ = 44100.0;
    int blockSize_ = 512;

    // Wavetables
    std::shared_ptr<Wavetable> currentWavetable_;
    std::shared_ptr<Wavetable> vowelWavetable_;

    // Polyphonic voices (8 voices)
    static constexpr int NUM_VOICES = 8;
    std::array<WavetableVoice, NUM_VOICES> voices_;
    std::array<FormantFilterBank, NUM_VOICES> formantFilters_;
    std::array<int, NUM_VOICES> voiceNotes_;  // MIDI note for each voice

    // Vibrato LFO
    float vibratoPhase_ = 0.0f;

    // Voice allocation
    int findFreeVoice();
    int findVoiceForNote(int midiNote);

    // Helpers
    static float midiToFrequency(int midiNote);
    static float centsToRatio(float cents);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(WavetableVoiceSynth)
};

//==============================================================================
/**
 * Wavetable preset factory
 */
class WavetablePresets
{
public:
    /** Get list of preset names */
    static juce::StringArray getPresetNames();

    /** Create wavetable from preset name */
    static std::shared_ptr<Wavetable> createPreset(const juce::String& name);

    /** Preset names */
    static constexpr const char* BASIC_SAW = "Basic Saw";
    static constexpr const char* BASIC_SQUARE = "Basic Square";
    static constexpr const char* BASIC_SINE = "Basic Sine";
    static constexpr const char* VOWEL_MORPH = "Vowel Morph";
    static constexpr const char* VOCAL_FORMANTS = "Vocal Formants";
    static constexpr const char* BRIGHT_VOCAL = "Bright Vocal";
    static constexpr const char* DARK_VOCAL = "Dark Vocal";
    static constexpr const char* CHOIR_PAD = "Choir Pad";
};

//==============================================================================
// Implementation notes:
//
// Wavetable synthesis + Formant filtering creates hybrid vocal sounds:
//
// 1. Wavetable provides the harmonic content (rich source signal)
// 2. Formant filters shape the timbre to sound like vowels
// 3. Wavetable morphing allows smooth transitions
// 4. Unison adds richness and width
//
// Signal flow:
//   Wavetable Osc → Formant Filter Bank → Envelope → Output
//        ↑                ↑
//   Position mod    Vowel morph
//
// This approach is lighter than pure formant synthesis while
// still achieving recognizable vocal timbres.
//==============================================================================
