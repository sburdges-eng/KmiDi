#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <array>
#include <vector>
#include <memory>
#include <atomic>

class BridgeClient;

//==============================================================================
/**
 * Formant data structure - matches Python FormantData
 */
struct FormantData
{
    float f1 = 730.0f;   // First formant (vowel height) Hz
    float f2 = 1090.0f;  // Second formant (vowel frontness) Hz
    float f3 = 2440.0f;  // Third formant (vowel rounding) Hz
    float bandwidth1 = 50.0f;
    float bandwidth2 = 70.0f;
    float bandwidth3 = 90.0f;
    float confidence = 1.0f;
};

//==============================================================================
/**
 * Vowel types matching Python VowelType enum
 */
enum class VowelType
{
    A,      // "ah" as in "father"
    E,      // "eh" as in "bed"
    I,      // "ee" as in "see"
    O,      // "oh" as in "go"
    U,      // "oo" as in "food"
    SCHWA,  // "uh" as in "about"
    Count
};

//==============================================================================
/**
 * Voice characteristics - matches Python VoiceCharacteristics
 */
struct VoiceCharacteristics
{
    // Formant data for each vowel
    std::array<FormantData, static_cast<size_t>(VowelType::Count)> vowelFormants;

    // Pitch characteristics
    float averagePitch = 200.0f;      // Hz
    float pitchRangeMin = 80.0f;      // Hz
    float pitchRangeMax = 400.0f;     // Hz
    float vibratoRate = 5.5f;         // Hz
    float vibratoDepth = 30.0f;       // cents

    // Timbre characteristics
    float spectralCentroid = 2000.0f;
    float spectralRolloff = 4000.0f;
    float spectralBandwidth = 1500.0f;

    // Voice quality
    float jitter = 0.5f;              // %
    float shimmer = 1.0f;             // %
    float breathiness = 0.1f;         // 0.0-1.0
    float nasality = 0.0f;            // 0.0-1.0

    // Attack/release
    float attackTime = 0.01f;         // seconds
    float releaseTime = 0.05f;        // seconds

    VoiceCharacteristics()
    {
        // Initialize standard formants for each vowel
        vowelFormants[static_cast<size_t>(VowelType::A)] = { 730, 1090, 2440, 50, 70, 90, 1.0f };
        vowelFormants[static_cast<size_t>(VowelType::E)] = { 570, 1980, 2440, 50, 70, 90, 1.0f };
        vowelFormants[static_cast<size_t>(VowelType::I)] = { 270, 2290, 3010, 50, 70, 90, 1.0f };
        vowelFormants[static_cast<size_t>(VowelType::O)] = { 570, 840, 2410, 50, 70, 90, 1.0f };
        vowelFormants[static_cast<size_t>(VowelType::U)] = { 300, 870, 2240, 50, 70, 90, 1.0f };
        vowelFormants[static_cast<size_t>(VowelType::SCHWA)] = { 500, 1500, 2500, 60, 80, 100, 1.0f };
    }
};

//==============================================================================
/**
 * Phoneme for synthesis queue
 */
struct SynthPhoneme
{
    VowelType vowelType = VowelType::A;
    float duration = 0.1f;          // seconds
    float pitch = 200.0f;           // Hz
    int stress = 0;                 // 0=unstressed, 1=primary, 2=secondary
    bool isConsonant = false;
    char consonantType = 0;         // For consonant synthesis
};

//==============================================================================
/**
 * Formant filter (2nd order IIR resonator)
 */
class FormantFilter
{
public:
    void setFormant(float frequency, float bandwidth, float sampleRate);
    float process(float input);
    void reset();

private:
    float a1 = 0.0f, a2 = 0.0f;
    float b0 = 0.0f;
    float x1 = 0.0f, x2 = 0.0f;
    float y1 = 0.0f, y2 = 0.0f;
};

//==============================================================================
/**
 * Glottal pulse generator (LF model)
 */
class GlottalSource
{
public:
    void setSampleRate(double sampleRate);
    void setFrequency(float frequency);
    void setParameters(float openQuotient, float returnQuotient);
    float process();
    void reset();

private:
    double sampleRate_ = 44100.0;
    float frequency_ = 200.0f;
    float phase_ = 0.0f;
    float openQuotient_ = 0.5f;    // Proportion of cycle glottis is open
    float returnQuotient_ = 0.1f;  // Proportion of cycle for return phase
};

//==============================================================================
/**
 * Real-time formant synthesizer voice
 */
class FormantSynthVoice
{
public:
    FormantSynthVoice();

    void setSampleRate(double sampleRate);
    void setVoiceCharacteristics(const VoiceCharacteristics& chars);
    void setCurrentVowel(VowelType vowel, float transitionTime = 0.05f);
    void setFrequency(float frequency);
    void noteOn(float velocity);
    void noteOff();

    float process();
    bool isActive() const { return active_; }

private:
    double sampleRate_ = 44100.0;
    bool active_ = false;

    // Voice characteristics
    VoiceCharacteristics voiceChars_;

    // Glottal source
    GlottalSource glottalSource_;

    // Formant filters (3 formants)
    FormantFilter formantFilters_[3];

    // Current and target formants (for interpolation)
    FormantData currentFormants_;
    FormantData targetFormants_;
    float formantTransitionRate_ = 0.0f;

    // Vibrato
    float vibratoPhase_ = 0.0f;

    // Jitter/shimmer
    juce::Random random_;

    // Envelope
    float envelope_ = 0.0f;
    float envelopeTarget_ = 0.0f;
    float attackRate_ = 0.0f;
    float releaseRate_ = 0.0f;

    // Breathiness noise
    float noiseLevel_ = 0.0f;

    void updateFormantFilters();
    float generateNoise();
};

//==============================================================================
/**
 * Full-featured voice processor with formant synthesis,
 * text-to-speech, and Python bridge integration.
 */
class VoiceProcessor : public juce::AudioProcessor
{
public:
    explicit VoiceProcessor(BridgeClient* client);
    ~VoiceProcessor() override = default;

    //==========================================================================
    // AudioProcessor overrides
    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    const juce::String getName() const override { return "DAiWVoiceProcessor"; }
    bool hasEditor() const override { return false; }
    juce::AudioProcessorEditor* createEditor() override { return nullptr; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}

    void getStateInformation(juce::MemoryBlock&) override {}
    void setStateInformation(const void*, int) override {}

    //==========================================================================
    // Voice synthesis API

    /** Load voice characteristics from Python model (via JSON) */
    bool loadVoiceModel(const juce::String& jsonData);

    /** Queue text for synthesis */
    void speakText(const juce::String& text);

    /** Queue phonemes directly */
    void queuePhonemes(const std::vector<SynthPhoneme>& phonemes);

    /** Set current vowel for real-time control */
    void setVowel(VowelType vowel);

    /** Set pitch in Hz */
    void setPitch(float pitch);

    /** Trigger note on (for MIDI control) */
    void noteOn(int midiNote, float velocity);

    /** Trigger note off */
    void noteOff();

    /** Get current voice characteristics */
    const VoiceCharacteristics& getVoiceCharacteristics() const { return voiceChars_; }

    /** Set voice characteristics */
    void setVoiceCharacteristics(const VoiceCharacteristics& chars);

    //==========================================================================
    // Modulation parameters (for real-time control)

    std::atomic<float> formantShift{ 1.0f };    // Formant frequency multiplier
    std::atomic<float> pitchShift{ 1.0f };      // Pitch multiplier
    std::atomic<float> breathiness{ 0.1f };     // Breath noise amount
    std::atomic<float> vibratoAmount{ 1.0f };   // Vibrato intensity

private:
    BridgeClient* bridgeClient_;
    double sampleRate_ = 44100.0;
    int blockSize_ = 512;

    // Voice characteristics
    VoiceCharacteristics voiceChars_;

    // Polyphonic voices (8 voices)
    static constexpr int NumVoices = 8;
    std::array<FormantSynthVoice, NumVoices> voices_;

    // Phoneme queue for TTS
    std::vector<SynthPhoneme> phonemeQueue_;
    size_t currentPhonemeIndex_ = 0;
    int samplesIntoCurrentPhoneme_ = 0;

    // Text-to-phoneme conversion
    std::vector<SynthPhoneme> textToPhonemes(const juce::String& text);

    // Find free voice for note
    int findFreeVoice();

    // Convert MIDI note to frequency
    static float midiToFrequency(int midiNote);

    // Parse JSON voice model
    bool parseVoiceModelJson(const juce::String& json);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(VoiceProcessor)
};
