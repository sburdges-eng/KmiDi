#include "VoiceProcessor.h"
#include "BridgeClient.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//==============================================================================
// FormantFilter Implementation
//==============================================================================

void FormantFilter::setFormant(float frequency, float bandwidth, float sampleRate)
{
    // Design a 2nd-order IIR resonator (bandpass filter)
    // Using bilinear transform of analog resonator

    const float omega = 2.0f * static_cast<float>(M_PI) * frequency / sampleRate;
    const float r = std::exp(-static_cast<float>(M_PI) * bandwidth / sampleRate);

    // Resonator coefficients
    a1 = -2.0f * r * std::cos(omega);
    a2 = r * r;

    // Normalize gain at resonant frequency
    b0 = (1.0f - r * r) * 0.5f;
}

float FormantFilter::process(float input)
{
    // Direct form II transposed
    float output = b0 * input - a1 * y1 - a2 * y2;

    y2 = y1;
    y1 = output;

    return output;
}

void FormantFilter::reset()
{
    x1 = x2 = y1 = y2 = 0.0f;
}

//==============================================================================
// GlottalSource Implementation (LF Model)
//==============================================================================

void GlottalSource::setSampleRate(double sampleRate)
{
    sampleRate_ = sampleRate;
}

void GlottalSource::setFrequency(float frequency)
{
    frequency_ = std::max(20.0f, std::min(frequency, 2000.0f));
}

void GlottalSource::setParameters(float openQuotient, float returnQuotient)
{
    openQuotient_ = std::max(0.3f, std::min(openQuotient, 0.8f));
    returnQuotient_ = std::max(0.05f, std::min(returnQuotient, 0.3f));
}

float GlottalSource::process()
{
    // LF (Liljencrants-Fant) glottal pulse model
    // Simplified version for real-time synthesis

    float output = 0.0f;

    // Opening phase (0 to openQuotient)
    if (phase_ < openQuotient_)
    {
        // Rising sinusoidal phase
        float t = phase_ / openQuotient_;
        output = 0.5f * (1.0f - std::cos(static_cast<float>(M_PI) * t));
    }
    // Closing phase (openQuotient to openQuotient + returnQuotient)
    else if (phase_ < openQuotient_ + returnQuotient_)
    {
        // Exponential decay (return phase)
        float t = (phase_ - openQuotient_) / returnQuotient_;
        output = std::exp(-5.0f * t);
    }
    // Closed phase (rest of cycle)
    else
    {
        output = 0.0f;
    }

    // Advance phase
    float phaseIncrement = frequency_ / static_cast<float>(sampleRate_);
    phase_ += phaseIncrement;
    if (phase_ >= 1.0f)
        phase_ -= 1.0f;

    // Differentiate to get glottal flow derivative (more natural sound)
    static float lastOutput = 0.0f;
    float derivative = output - lastOutput;
    lastOutput = output;

    return derivative * 2.0f;
}

void GlottalSource::reset()
{
    phase_ = 0.0f;
}

//==============================================================================
// FormantSynthVoice Implementation
//==============================================================================

FormantSynthVoice::FormantSynthVoice()
{
    currentFormants_ = voiceChars_.vowelFormants[static_cast<size_t>(VowelType::A)];
    targetFormants_ = currentFormants_;
}

void FormantSynthVoice::setSampleRate(double sampleRate)
{
    sampleRate_ = sampleRate;
    glottalSource_.setSampleRate(sampleRate);

    // Calculate envelope rates
    attackRate_ = 1.0f / (voiceChars_.attackTime * static_cast<float>(sampleRate));
    releaseRate_ = 1.0f / (voiceChars_.releaseTime * static_cast<float>(sampleRate));

    updateFormantFilters();
}

void FormantSynthVoice::setVoiceCharacteristics(const VoiceCharacteristics &chars)
{
    voiceChars_ = chars;
    attackRate_ = 1.0f / (chars.attackTime * static_cast<float>(sampleRate_));
    releaseRate_ = 1.0f / (chars.releaseTime * static_cast<float>(sampleRate_));
}

void FormantSynthVoice::setCurrentVowel(VowelType vowel, float transitionTime)
{
    targetFormants_ = voiceChars_.vowelFormants[static_cast<size_t>(vowel)];

    if (transitionTime > 0.0f)
    {
        formantTransitionRate_ = 1.0f / (transitionTime * static_cast<float>(sampleRate_));
    }
    else
    {
        currentFormants_ = targetFormants_;
        formantTransitionRate_ = 0.0f;
        updateFormantFilters();
    }
}

void FormantSynthVoice::setFrequency(float frequency)
{
    glottalSource_.setFrequency(frequency);
}

void FormantSynthVoice::noteOn(float velocity)
{
    active_ = true;
    envelopeTarget_ = velocity;
    noiseLevel_ = voiceChars_.breathiness;

    // Reset filters for clean attack
    for (auto &filter : formantFilters_)
        filter.reset();

    glottalSource_.reset();
}

void FormantSynthVoice::noteOff()
{
    envelopeTarget_ = 0.0f;
}

float FormantSynthVoice::process()
{
    if (!active_)
        return 0.0f;

    // Update envelope
    if (envelope_ < envelopeTarget_)
    {
        envelope_ += attackRate_;
        if (envelope_ > envelopeTarget_)
            envelope_ = envelopeTarget_;
    }
    else if (envelope_ > envelopeTarget_)
    {
        envelope_ -= releaseRate_;
        if (envelope_ < 0.0f)
        {
            envelope_ = 0.0f;
            active_ = false;
            return 0.0f;
        }
    }

    // Interpolate formants if transitioning
    if (formantTransitionRate_ > 0.0f)
    {
        bool needsUpdate = false;

        auto interpolate = [&](float &current, float target)
        {
            if (std::abs(current - target) > 1.0f)
            {
                current += (target - current) * formantTransitionRate_;
                needsUpdate = true;
            }
            else
            {
                current = target;
            }
        };

        interpolate(currentFormants_.f1, targetFormants_.f1);
        interpolate(currentFormants_.f2, targetFormants_.f2);
        interpolate(currentFormants_.f3, targetFormants_.f3);

        if (needsUpdate)
            updateFormantFilters();
        else
            formantTransitionRate_ = 0.0f;
    }

    // Apply vibrato
    float vibratoMod = 1.0f;
    if (voiceChars_.vibratoRate > 0.0f && voiceChars_.vibratoDepth > 0.0f)
    {
        vibratoPhase_ += voiceChars_.vibratoRate / static_cast<float>(sampleRate_);
        if (vibratoPhase_ >= 1.0f)
            vibratoPhase_ -= 1.0f;

        // Convert cents to frequency ratio
        float vibratoCents = voiceChars_.vibratoDepth * std::sin(2.0f * static_cast<float>(M_PI) * vibratoPhase_);
        vibratoMod = std::pow(2.0f, vibratoCents / 1200.0f);
    }

    // Apply jitter (pitch perturbation)
    float jitterMod = 1.0f;
    if (voiceChars_.jitter > 0.0f)
    {
        jitterMod = 1.0f + (random_.nextFloat() - 0.5f) * voiceChars_.jitter * 0.02f;
    }

    // Generate glottal source with modulation
    float currentFreq = glottalSource_.process();

    // Apply formant filters (parallel configuration)
    float f1Out = formantFilters_[0].process(currentFreq);
    float f2Out = formantFilters_[1].process(currentFreq);
    float f3Out = formantFilters_[2].process(currentFreq);

    // Mix formants (F1 dominant, F2 and F3 for timbre)
    float formantMix = f1Out * 1.0f + f2Out * 0.7f + f3Out * 0.4f;

    // Add breathiness (aspiration noise)
    float output = formantMix;
    if (noiseLevel_ > 0.0f)
    {
        float noise = generateNoise();
        // High-pass filter noise for aspiration
        output = output * (1.0f - noiseLevel_ * 0.5f) + noise * noiseLevel_;
    }

    // Apply shimmer (amplitude perturbation)
    float shimmerMod = 1.0f;
    if (voiceChars_.shimmer > 0.0f)
    {
        shimmerMod = 1.0f + (random_.nextFloat() - 0.5f) * voiceChars_.shimmer * 0.02f;
    }

    // Apply envelope and modulation
    output *= envelope_ * shimmerMod;

    return output;
}

void FormantSynthVoice::updateFormantFilters()
{
    formantFilters_[0].setFormant(currentFormants_.f1, currentFormants_.bandwidth1, static_cast<float>(sampleRate_));
    formantFilters_[1].setFormant(currentFormants_.f2, currentFormants_.bandwidth2, static_cast<float>(sampleRate_));
    formantFilters_[2].setFormant(currentFormants_.f3, currentFormants_.bandwidth3, static_cast<float>(sampleRate_));
}

float FormantSynthVoice::generateNoise()
{
    // Simple white noise
    return (random_.nextFloat() * 2.0f - 1.0f) * 0.3f;
}

//==============================================================================
// VoiceProcessor Implementation
//==============================================================================

VoiceProcessor::VoiceProcessor(BridgeClient *client)
    : bridgeClient_(client)
{
}

void VoiceProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    sampleRate_ = sampleRate;
    blockSize_ = samplesPerBlock;

    // Initialize all voices
    for (auto &voice : voices_)
    {
        voice.setSampleRate(sampleRate);
        voice.setVoiceCharacteristics(voiceChars_);
    }
}

void VoiceProcessor::releaseResources()
{
    phonemeQueue_.clear();
    currentPhonemeIndex_ = 0;
    samplesIntoCurrentPhoneme_ = 0;
}

void VoiceProcessor::processBlock(juce::AudioBuffer<float> &buffer, juce::MidiBuffer &midiMessages)
{
    // Handle MIDI messages
    for (const auto metadata : midiMessages)
    {
        auto message = metadata.getMessage();

        if (message.isNoteOn())
        {
            noteOn(message.getNoteNumber(), message.getFloatVelocity());
        }
        else if (message.isNoteOff())
        {
            noteOff();
        }
    }

    // Process audio
    auto *leftChannel = buffer.getWritePointer(0);
    auto *rightChannel = buffer.getNumChannels() > 1 ? buffer.getWritePointer(1) : nullptr;

    for (int sample = 0; sample < buffer.getNumSamples(); ++sample)
    {
        float outputSample = 0.0f;

        // Process TTS phoneme queue if active
        if (currentPhonemeIndex_ < phonemeQueue_.size())
        {
            const auto &currentPhoneme = phonemeQueue_[currentPhonemeIndex_];
            int phonemeSamples = static_cast<int>(currentPhoneme.duration * sampleRate_);

            // Set vowel for first voice (TTS voice)
            if (samplesIntoCurrentPhoneme_ == 0)
            {
                if (!currentPhoneme.isConsonant)
                {
                    voices_[0].setCurrentVowel(currentPhoneme.vowelType, 0.02f);
                    voices_[0].setFrequency(currentPhoneme.pitch * pitchShift.load());

                    if (!voices_[0].isActive())
                        voices_[0].noteOn(0.8f);
                }
            }

            samplesIntoCurrentPhoneme_++;

            if (samplesIntoCurrentPhoneme_ >= phonemeSamples)
            {
                samplesIntoCurrentPhoneme_ = 0;
                currentPhonemeIndex_++;

                // Note off at end of queue
                if (currentPhonemeIndex_ >= phonemeQueue_.size())
                {
                    voices_[0].noteOff();
                }
            }
        }

        // Sum all active voices
        for (auto &voice : voices_)
        {
            if (voice.isActive())
            {
                outputSample += voice.process();
            }
        }

        // Apply global modulation
        outputSample *= 0.5f; // Master gain

        // Soft clipping
        outputSample = std::tanh(outputSample);

        // Write to output
        leftChannel[sample] = outputSample;
        if (rightChannel)
            rightChannel[sample] = outputSample;
    }
}

bool VoiceProcessor::loadVoiceModel(const juce::String &jsonData)
{
    return parseVoiceModelJson(jsonData);
}

void VoiceProcessor::speakText(const juce::String &text)
{
    auto phonemes = textToPhonemes(text);
    queuePhonemes(phonemes);
}

void VoiceProcessor::queuePhonemes(const std::vector<SynthPhoneme> &phonemes)
{
    phonemeQueue_ = phonemes;
    currentPhonemeIndex_ = 0;
    samplesIntoCurrentPhoneme_ = 0;
}

void VoiceProcessor::setVowel(VowelType vowel)
{
    // Set vowel on first available voice
    int voiceIndex = findFreeVoice();
    if (voiceIndex >= 0)
    {
        voices_[voiceIndex].setCurrentVowel(vowel);
    }
}

void VoiceProcessor::setPitch(float pitch)
{
    for (auto &voice : voices_)
    {
        if (voice.isActive())
        {
            voice.setFrequency(pitch * pitchShift.load());
        }
    }
}

void VoiceProcessor::noteOn(int midiNote, float velocity)
{
    int voiceIndex = findFreeVoice();
    if (voiceIndex >= 0)
    {
        voices_[voiceIndex].setFrequency(midiToFrequency(midiNote) * pitchShift.load());
        voices_[voiceIndex].noteOn(velocity);
    }
}

void VoiceProcessor::noteOff()
{
    // Note off all voices (simple implementation)
    for (auto &voice : voices_)
    {
        if (voice.isActive())
        {
            voice.noteOff();
        }
    }
}

void VoiceProcessor::setVoiceCharacteristics(const VoiceCharacteristics &chars)
{
    voiceChars_ = chars;
    for (auto &voice : voices_)
    {
        voice.setVoiceCharacteristics(chars);
    }
}

int VoiceProcessor::findFreeVoice()
{
    // Find inactive voice
    for (int i = 0; i < NumVoices; ++i)
    {
        if (!voices_[i].isActive())
            return i;
    }
    // Steal oldest voice (voice 0)
    return 0;
}

float VoiceProcessor::midiToFrequency(int midiNote)
{
    return 440.0f * std::pow(2.0f, (midiNote - 69) / 12.0f);
}

std::vector<SynthPhoneme> VoiceProcessor::textToPhonemes(const juce::String &text)
{
    // Simple text-to-phoneme conversion
    // In production, this would use a proper phoneme dictionary or call Python

    std::vector<SynthPhoneme> phonemes;
    juce::String normalized = text.toLowerCase().trim();

    // Simple vowel mapping
    auto charToVowel = [](juce::juce_wchar c) -> VowelType
    {
        switch (c)
        {
        case 'a':
            return VowelType::A;
        case 'e':
            return VowelType::E;
        case 'i':
            return VowelType::I;
        case 'o':
            return VowelType::O;
        case 'u':
            return VowelType::U;
        default:
            return VowelType::SCHWA;
        }
    };

    float basePitch = voiceChars_.averagePitch;
    bool lastWasVowel = false;

    for (int i = 0; i < normalized.length(); ++i)
    {
        juce::juce_wchar c = normalized[i];

        if (c == ' ')
        {
            // Add silence between words
            SynthPhoneme silence;
            silence.isConsonant = true;
            silence.duration = 0.1f;
            phonemes.push_back(silence);
            lastWasVowel = false;
        }
        else if (juce::String("aeiou").containsChar(c))
        {
            SynthPhoneme vowel;
            vowel.vowelType = charToVowel(c);
            vowel.duration = 0.12f;
            vowel.pitch = basePitch;
            vowel.stress = lastWasVowel ? 0 : 1; // Primary stress on first vowel
            vowel.isConsonant = false;
            phonemes.push_back(vowel);
            lastWasVowel = true;
        }
        else if (juce::CharacterFunctions::isLetter(c))
        {
            // Consonant - brief noise or transition
            SynthPhoneme consonant;
            consonant.isConsonant = true;
            consonant.consonantType = static_cast<char>(c);
            consonant.duration = 0.06f;
            phonemes.push_back(consonant);
            lastWasVowel = false;
        }
    }

    return phonemes;
}

bool VoiceProcessor::parseVoiceModelJson(const juce::String &json)
{
    // Parse JSON voice model from Python
    // Expected format matches Python VoiceCharacteristics

    auto result = juce::JSON::parse(json);
    if (result.isVoid())
        return false;

    auto *obj = result.getDynamicObject();
    if (!obj)
        return false;

    auto &chars = obj->getProperties();

    // Parse pitch characteristics
    if (chars.contains("average_pitch"))
        voiceChars_.averagePitch = static_cast<float>(chars["average_pitch"]);

    if (chars.contains("pitch_range"))
    {
        auto pitchRange = chars["pitch_range"];
        if (pitchRange.isArray() && pitchRange.getArray()->size() >= 2)
        {
            voiceChars_.pitchRangeMin = static_cast<float>((*pitchRange.getArray())[0]);
            voiceChars_.pitchRangeMax = static_cast<float>((*pitchRange.getArray())[1]);
        }
    }

    if (chars.contains("vibrato_rate"))
        voiceChars_.vibratoRate = static_cast<float>(chars["vibrato_rate"]);

    if (chars.contains("vibrato_depth"))
        voiceChars_.vibratoDepth = static_cast<float>(chars["vibrato_depth"]);

    // Parse voice quality
    if (chars.contains("jitter"))
        voiceChars_.jitter = static_cast<float>(chars["jitter"]);

    if (chars.contains("shimmer"))
        voiceChars_.shimmer = static_cast<float>(chars["shimmer"]);

    if (chars.contains("breathiness"))
        voiceChars_.breathiness = static_cast<float>(chars["breathiness"]);

    if (chars.contains("nasality"))
        voiceChars_.nasality = static_cast<float>(chars["nasality"]);

    // Parse timbre
    if (chars.contains("spectral_centroid_mean"))
        voiceChars_.spectralCentroid = static_cast<float>(chars["spectral_centroid_mean"]);

    if (chars.contains("spectral_rolloff_mean"))
        voiceChars_.spectralRolloff = static_cast<float>(chars["spectral_rolloff_mean"]);

    // Parse vowel formants
    if (chars.contains("vowel_formants"))
    {
        auto vowelFormants = chars["vowel_formants"].getDynamicObject();
        if (vowelFormants)
        {
            auto parseFormants = [this](const juce::String &vowelName, VowelType type, juce::DynamicObject *obj)
            {
                if (obj->hasProperty(vowelName))
                {
                    auto formantArray = obj->getProperty(vowelName);
                    if (formantArray.isArray() && formantArray.getArray()->size() > 0)
                    {
                        auto firstFormant = (*formantArray.getArray())[0].getDynamicObject();
                        if (firstFormant)
                        {
                            auto &formants = voiceChars_.vowelFormants[static_cast<size_t>(type)];
                            if (firstFormant->hasProperty("f1"))
                                formants.f1 = static_cast<float>(firstFormant->getProperty("f1"));
                            if (firstFormant->hasProperty("f2"))
                                formants.f2 = static_cast<float>(firstFormant->getProperty("f2"));
                            if (firstFormant->hasProperty("f3"))
                                formants.f3 = static_cast<float>(firstFormant->getProperty("f3"));
                        }
                    }
                }
            };

            parseFormants("a", VowelType::A, vowelFormants);
            parseFormants("e", VowelType::E, vowelFormants);
            parseFormants("i", VowelType::I, vowelFormants);
            parseFormants("o", VowelType::O, vowelFormants);
            parseFormants("u", VowelType::U, vowelFormants);
            parseFormants("ə", VowelType::SCHWA, vowelFormants);
        }
    }

    // Update all voices with new characteristics
    setVoiceCharacteristics(voiceChars_);

    return true;
}
