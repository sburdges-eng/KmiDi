#include "WavetableSynth.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//==============================================================================
// Wavetable Implementation
//==============================================================================

Wavetable::Wavetable()
{
    frames_.resize(1);
    generateSine();
}

void Wavetable::generateSine()
{
    frames_.resize(1);
    numFrames_ = 1;

    for (int i = 0; i < WAVETABLE_SIZE; ++i)
    {
        float phase = static_cast<float>(i) / WAVETABLE_SIZE;
        frames_[0][i] = std::sin(2.0f * static_cast<float>(M_PI) * phase);
    }
}

void Wavetable::generateSaw()
{
    frames_.resize(1);
    numFrames_ = 1;

    // Band-limited sawtooth using additive synthesis
    for (int i = 0; i < WAVETABLE_SIZE; ++i)
    {
        float phase = static_cast<float>(i) / WAVETABLE_SIZE;
        float sample = 0.0f;

        // Add harmonics (limit to avoid aliasing)
        for (int h = 1; h <= 32; ++h)
        {
            sample += std::sin(2.0f * static_cast<float>(M_PI) * h * phase) / h;
        }

        frames_[0][i] = sample * 0.5f;
    }
}

void Wavetable::generateSquare()
{
    frames_.resize(1);
    numFrames_ = 1;

    // Band-limited square using odd harmonics
    for (int i = 0; i < WAVETABLE_SIZE; ++i)
    {
        float phase = static_cast<float>(i) / WAVETABLE_SIZE;
        float sample = 0.0f;

        for (int h = 1; h <= 31; h += 2)  // Odd harmonics only
        {
            sample += std::sin(2.0f * static_cast<float>(M_PI) * h * phase) / h;
        }

        frames_[0][i] = sample * 0.6f;
    }
}

void Wavetable::generateTriangle()
{
    frames_.resize(1);
    numFrames_ = 1;

    // Band-limited triangle
    for (int i = 0; i < WAVETABLE_SIZE; ++i)
    {
        float phase = static_cast<float>(i) / WAVETABLE_SIZE;
        float sample = 0.0f;
        int sign = 1;

        for (int h = 1; h <= 31; h += 2)
        {
            sample += sign * std::sin(2.0f * static_cast<float>(M_PI) * h * phase) / (h * h);
            sign = -sign;
        }

        frames_[0][i] = sample * 1.2f;
    }
}

void Wavetable::generatePulse(float width)
{
    frames_.resize(1);
    numFrames_ = 1;

    width = juce::jlimit(0.1f, 0.9f, width);

    for (int i = 0; i < WAVETABLE_SIZE; ++i)
    {
        float phase = static_cast<float>(i) / WAVETABLE_SIZE;
        frames_[0][i] = phase < width ? 0.8f : -0.8f;
    }
}

void Wavetable::generateVowelFormants(const std::array<float, 5>& formantFreqs)
{
    frames_.resize(1);
    numFrames_ = 1;

    // Create wavetable by summing formant-weighted harmonics
    const float baseFreq = 100.0f;  // Assumed fundamental

    for (int i = 0; i < WAVETABLE_SIZE; ++i)
    {
        float phase = static_cast<float>(i) / WAVETABLE_SIZE;
        float sample = 0.0f;

        // Add harmonics with formant-based amplitudes
        for (int h = 1; h <= 64; ++h)
        {
            float harmFreq = baseFreq * h;
            float amplitude = 0.0f;

            // Calculate amplitude based on proximity to formants
            for (int f = 0; f < 5; ++f)
            {
                float dist = std::abs(harmFreq - formantFreqs[f]);
                float bandwidth = 100.0f;  // Formant bandwidth
                float gain = std::exp(-dist * dist / (2.0f * bandwidth * bandwidth));
                amplitude += gain * (1.0f / (f + 1));  // Higher formants weaker
            }

            sample += amplitude * std::sin(2.0f * static_cast<float>(M_PI) * h * phase) / h;
        }

        frames_[0][i] = sample * 0.5f;
    }
}

void Wavetable::generateVowelMorph()
{
    // Create 5 frames, one for each vowel
    numFrames_ = 5;
    frames_.resize(5);

    // Standard vowel formants (F1, F2, F3, F4, F5)
    const std::array<std::array<float, 5>, 5> vowelFormants = {{
        {{ 730, 1090, 2440, 3400, 4500 }},  // A
        {{ 570, 1980, 2440, 3400, 4500 }},  // E
        {{ 270, 2290, 3010, 3400, 4500 }},  // I
        {{ 570,  840, 2410, 3400, 4500 }},  // O
        {{ 300,  870, 2240, 3400, 4500 }}   // U
    }};

    const float baseFreq = 100.0f;

    for (int v = 0; v < 5; ++v)
    {
        for (int i = 0; i < WAVETABLE_SIZE; ++i)
        {
            float phase = static_cast<float>(i) / WAVETABLE_SIZE;
            float sample = 0.0f;

            for (int h = 1; h <= 64; ++h)
            {
                float harmFreq = baseFreq * h;
                float amplitude = 0.0f;

                for (int f = 0; f < 5; ++f)
                {
                    float dist = std::abs(harmFreq - vowelFormants[v][f]);
                    float bandwidth = 80.0f + f * 20.0f;
                    float gain = std::exp(-dist * dist / (2.0f * bandwidth * bandwidth));
                    amplitude += gain * (1.0f / (f + 1));
                }

                sample += amplitude * std::sin(2.0f * static_cast<float>(M_PI) * h * phase) / h;
            }

            frames_[v][i] = sample * 0.5f;
        }
    }
}

void Wavetable::loadFromData(const float* data, int numFrames, int samplesPerFrame)
{
    numFrames_ = std::min(numFrames, MAX_WAVETABLE_FRAMES);
    frames_.resize(numFrames_);

    for (int f = 0; f < numFrames_; ++f)
    {
        for (int i = 0; i < WAVETABLE_SIZE; ++i)
        {
            int srcIdx = (i * samplesPerFrame) / WAVETABLE_SIZE;
            srcIdx = std::min(srcIdx, samplesPerFrame - 1);
            frames_[f][i] = data[f * samplesPerFrame + srcIdx];
        }
    }
}

float Wavetable::getSample(float phase, float framePosition) const
{
    // Clamp frame position
    framePosition = juce::jlimit(0.0f, static_cast<float>(numFrames_ - 1), framePosition);

    if (numFrames_ == 1)
    {
        return interpolateFrame(frames_[0], phase);
    }

    // Interpolate between frames
    int frame1 = static_cast<int>(framePosition);
    int frame2 = std::min(frame1 + 1, numFrames_ - 1);
    float frameFrac = framePosition - frame1;

    float sample1 = interpolateFrame(frames_[frame1], phase);
    float sample2 = interpolateFrame(frames_[frame2], phase);

    return sample1 + frameFrac * (sample2 - sample1);
}

float Wavetable::interpolateFrame(const std::array<float, WAVETABLE_SIZE>& frame, float phase) const
{
    // Wrap phase
    phase = phase - std::floor(phase);

    float indexFloat = phase * WAVETABLE_SIZE;
    int index1 = static_cast<int>(indexFloat) % WAVETABLE_SIZE;
    int index2 = (index1 + 1) % WAVETABLE_SIZE;
    float frac = indexFloat - std::floor(indexFloat);

    // Linear interpolation
    return frame[index1] + frac * (frame[index2] - frame[index1]);
}

//==============================================================================
// FormantFilterBank Implementation
//==============================================================================

FormantFilterBank::FormantFilterBank()
{
    // Initialize with 'A' vowel
    setVowel(0);

    // Default gains (formant 1 strongest)
    gains_ = {{ 1.0f, 0.6f, 0.3f, 0.15f, 0.1f }};
}

void FormantFilterBank::setSampleRate(double sampleRate)
{
    sampleRate_ = sampleRate;
}

void FormantFilterBank::BiquadFilter::setResonator(float frequency, float bandwidth, float sampleRate)
{
    const float omega = 2.0f * static_cast<float>(M_PI) * frequency / static_cast<float>(sampleRate);
    const float r = std::exp(-static_cast<float>(M_PI) * bandwidth / static_cast<float>(sampleRate));

    a1 = -2.0f * r * std::cos(omega);
    a2 = r * r;

    // Bandpass resonator coefficients
    b0 = (1.0f - r * r) * 0.5f;
    b1 = 0.0f;
    b2 = -b0;
}

float FormantFilterBank::BiquadFilter::process(float input)
{
    float output = b0 * input + b1 * z1 + b2 * z2 - a1 * z1 - a2 * z2;

    z2 = z1;
    z1 = output;

    return output;
}

void FormantFilterBank::setFormants(const std::array<float, 5>& frequencies,
                                    const std::array<float, 5>& bandwidths)
{
    for (int i = 0; i < NUM_FORMANT_FILTERS; ++i)
    {
        filters_[i].setResonator(frequencies[i], bandwidths[i], static_cast<float>(sampleRate_));
    }
}

void FormantFilterBank::setVowel(int vowelIndex)
{
    vowelIndex = juce::jlimit(0, 4, vowelIndex);

    std::array<float, 5> bandwidths = {{ 50, 70, 90, 100, 120 }};

    for (int i = 0; i < NUM_FORMANT_FILTERS; ++i)
    {
        filters_[i].setResonator(
            VOWEL_FORMANTS[vowelIndex][i],
            bandwidths[i],
            static_cast<float>(sampleRate_)
        );
    }
}

void FormantFilterBank::morphVowels(int vowel1, int vowel2, float position)
{
    vowel1 = juce::jlimit(0, 4, vowel1);
    vowel2 = juce::jlimit(0, 4, vowel2);
    position = juce::jlimit(0.0f, 1.0f, position);

    std::array<float, 5> bandwidths = {{ 50, 70, 90, 100, 120 }};

    for (int i = 0; i < NUM_FORMANT_FILTERS; ++i)
    {
        float freq = VOWEL_FORMANTS[vowel1][i] * (1.0f - position) +
                     VOWEL_FORMANTS[vowel2][i] * position;

        filters_[i].setResonator(freq, bandwidths[i], static_cast<float>(sampleRate_));
    }
}

float FormantFilterBank::process(float input)
{
    float output = 0.0f;

    for (int i = 0; i < NUM_FORMANT_FILTERS; ++i)
    {
        output += filters_[i].process(input) * gains_[i];
    }

    return output;
}

void FormantFilterBank::reset()
{
    for (auto& filter : filters_)
    {
        filter.reset();
    }
}

//==============================================================================
// WavetableVoice Implementation
//==============================================================================

WavetableVoice::WavetableVoice()
{
    phases_.fill(0.0f);
    detuneFactors_.fill(1.0f);
}

void WavetableVoice::setSampleRate(double sampleRate)
{
    sampleRate_ = sampleRate;
}

void WavetableVoice::setWavetable(std::shared_ptr<Wavetable> wavetable)
{
    wavetable_ = wavetable;
}

void WavetableVoice::setFrequency(float frequency)
{
    frequency_ = frequency;
}

void WavetableVoice::setWavetablePosition(float position)
{
    wavetablePos_ = position;
}

void WavetableVoice::setUnison(int numVoices, float detune)
{
    unisonVoices_ = juce::jlimit(1, MAX_UNISON_VOICES, numVoices);
    unisonDetune_ = detune;
    updateDetuneFactors();
}

void WavetableVoice::updateDetuneFactors()
{
    if (unisonVoices_ == 1)
    {
        detuneFactors_[0] = 1.0f;
        return;
    }

    // Spread detune across voices
    for (int i = 0; i < unisonVoices_; ++i)
    {
        float spread = static_cast<float>(i) / (unisonVoices_ - 1) * 2.0f - 1.0f;  // -1 to 1
        float cents = spread * unisonDetune_;
        detuneFactors_[i] = std::pow(2.0f, cents / 1200.0f);
    }
}

void WavetableVoice::noteOn(float velocity)
{
    active_ = true;
    envelopeTarget_ = velocity;

    // Randomize phases for natural sound
    juce::Random random;
    for (int i = 0; i < unisonVoices_; ++i)
    {
        phases_[i] = random.nextFloat();
    }
}

void WavetableVoice::noteOff()
{
    envelopeTarget_ = 0.0f;
}

float WavetableVoice::process()
{
    if (!active_ || !wavetable_)
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
        if (envelope_ < 0.0001f)
        {
            envelope_ = 0.0f;
            active_ = false;
            return 0.0f;
        }
    }

    // Sum unison voices
    float output = 0.0f;
    float framePos = wavetablePos_ * (wavetable_->getNumFrames() - 1);

    for (int i = 0; i < unisonVoices_; ++i)
    {
        output += wavetable_->getSample(phases_[i], framePos);

        // Advance phase
        float freq = frequency_ * detuneFactors_[i];
        phases_[i] += freq / static_cast<float>(sampleRate_);
        if (phases_[i] >= 1.0f)
            phases_[i] -= 1.0f;
    }

    // Normalize for unison
    output /= std::sqrt(static_cast<float>(unisonVoices_));

    return output * envelope_;
}

//==============================================================================
// WavetableVoiceSynth Implementation
//==============================================================================

WavetableVoiceSynth::WavetableVoiceSynth()
{
    voiceNotes_.fill(-1);

    // Create default wavetables
    currentWavetable_ = std::make_shared<Wavetable>();
    currentWavetable_->generateSaw();

    vowelWavetable_ = std::make_shared<Wavetable>();
    vowelWavetable_->generateVowelMorph();
}

void WavetableVoiceSynth::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    sampleRate_ = sampleRate;
    blockSize_ = samplesPerBlock;

    for (auto& voice : voices_)
    {
        voice.setSampleRate(sampleRate);
        voice.setWavetable(currentWavetable_);
    }

    for (auto& filter : formantFilters_)
    {
        filter.setSampleRate(sampleRate);
    }
}

void WavetableVoiceSynth::releaseResources()
{
    for (auto& filter : formantFilters_)
    {
        filter.reset();
    }
}

void WavetableVoiceSynth::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    // Handle MIDI
    for (const auto metadata : midiMessages)
    {
        auto message = metadata.getMessage();

        if (message.isNoteOn())
        {
            noteOn(message.getNoteNumber(), message.getFloatVelocity());
        }
        else if (message.isNoteOff())
        {
            int note = message.getNoteNumber();
            int voiceIdx = findVoiceForNote(note);
            if (voiceIdx >= 0)
            {
                voices_[voiceIdx].noteOff();
                voiceNotes_[voiceIdx] = -1;
            }
        }
    }

    // Get parameters
    float wtPos = wavetablePosition.load();
    float vowelPos = vowelPosition.load();
    float fmtMix = formantMix.load();
    int unison = unisonVoices.load();
    float detune = unisonDetune.load();
    float vibRate = vibratoRate.load();
    float vibDepth = vibratoDepth.load();
    float gain = std::pow(10.0f, outputGain.load() / 20.0f);

    // Update voice parameters
    for (int v = 0; v < NUM_VOICES; ++v)
    {
        voices_[v].setWavetablePosition(wtPos);
        voices_[v].setUnison(unison, detune);

        // Update formant filter for vowel position
        int vowel1 = static_cast<int>(vowelPos);
        int vowel2 = std::min(vowel1 + 1, 4);
        float vowelFrac = vowelPos - vowel1;
        formantFilters_[v].morphVowels(vowel1, vowel2, vowelFrac);
    }

    // Process audio
    auto* leftChannel = buffer.getWritePointer(0);
    auto* rightChannel = buffer.getNumChannels() > 1 ? buffer.getWritePointer(1) : nullptr;

    for (int sample = 0; sample < buffer.getNumSamples(); ++sample)
    {
        // Update vibrato LFO
        vibratoPhase_ += vibRate / static_cast<float>(sampleRate_);
        if (vibratoPhase_ >= 1.0f)
            vibratoPhase_ -= 1.0f;

        float vibratoMod = std::sin(2.0f * static_cast<float>(M_PI) * vibratoPhase_);
        float vibratoRatio = std::pow(2.0f, vibratoMod * vibDepth / 1200.0f);

        float outputSample = 0.0f;

        for (int v = 0; v < NUM_VOICES; ++v)
        {
            if (voices_[v].isActive())
            {
                // Get wavetable output
                float voiceSample = voices_[v].process();

                // Apply formant filter
                float filtered = formantFilters_[v].process(voiceSample);

                // Mix dry/wet
                voiceSample = voiceSample * (1.0f - fmtMix) + filtered * fmtMix;

                outputSample += voiceSample;
            }
        }

        // Apply gain and soft clip
        outputSample *= gain;
        outputSample = std::tanh(outputSample);

        leftChannel[sample] = outputSample;
        if (rightChannel)
            rightChannel[sample] = outputSample;
    }
}

void WavetableVoiceSynth::noteOn(int midiNote, float velocity)
{
    int voiceIdx = findFreeVoice();

    voices_[voiceIdx].setFrequency(midiToFrequency(midiNote));
    voices_[voiceIdx].noteOn(velocity);
    voiceNotes_[voiceIdx] = midiNote;

    formantFilters_[voiceIdx].reset();
}

void WavetableVoiceSynth::noteOff()
{
    for (auto& voice : voices_)
    {
        voice.noteOff();
    }
    voiceNotes_.fill(-1);
}

void WavetableVoiceSynth::setVowel(int vowelIndex)
{
    vowelPosition.store(static_cast<float>(vowelIndex));
}

void WavetableVoiceSynth::loadWavetable(const float* data, int numFrames, int samplesPerFrame)
{
    auto newWT = std::make_shared<Wavetable>();
    newWT->loadFromData(data, numFrames, samplesPerFrame);
    currentWavetable_ = newWT;

    for (auto& voice : voices_)
    {
        voice.setWavetable(currentWavetable_);
    }
}

void WavetableVoiceSynth::loadPresetWavetable(const juce::String& presetName)
{
    auto newWT = WavetablePresets::createPreset(presetName);
    if (newWT)
    {
        currentWavetable_ = newWT;
        for (auto& voice : voices_)
        {
            voice.setWavetable(currentWavetable_);
        }
    }
}

int WavetableVoiceSynth::findFreeVoice()
{
    // Find inactive voice
    for (int i = 0; i < NUM_VOICES; ++i)
    {
        if (!voices_[i].isActive())
            return i;
    }
    // Steal voice 0
    return 0;
}

int WavetableVoiceSynth::findVoiceForNote(int midiNote)
{
    for (int i = 0; i < NUM_VOICES; ++i)
    {
        if (voiceNotes_[i] == midiNote)
            return i;
    }
    return -1;
}

float WavetableVoiceSynth::midiToFrequency(int midiNote)
{
    return 440.0f * std::pow(2.0f, (midiNote - 69) / 12.0f);
}

float WavetableVoiceSynth::centsToRatio(float cents)
{
    return std::pow(2.0f, cents / 1200.0f);
}

//==============================================================================
// WavetablePresets Implementation
//==============================================================================

juce::StringArray WavetablePresets::getPresetNames()
{
    return {
        BASIC_SAW,
        BASIC_SQUARE,
        BASIC_SINE,
        VOWEL_MORPH,
        VOCAL_FORMANTS,
        BRIGHT_VOCAL,
        DARK_VOCAL,
        CHOIR_PAD
    };
}

std::shared_ptr<Wavetable> WavetablePresets::createPreset(const juce::String& name)
{
    auto wt = std::make_shared<Wavetable>();

    if (name == BASIC_SAW)
        wt->generateSaw();
    else if (name == BASIC_SQUARE)
        wt->generateSquare();
    else if (name == BASIC_SINE)
        wt->generateSine();
    else if (name == VOWEL_MORPH)
        wt->generateVowelMorph();
    else if (name == VOCAL_FORMANTS)
        wt->generateVowelFormants({{ 730, 1090, 2440, 3400, 4500 }});
    else if (name == BRIGHT_VOCAL)
        wt->generateVowelFormants({{ 400, 2000, 2800, 3500, 4800 }});
    else if (name == DARK_VOCAL)
        wt->generateVowelFormants({{ 500, 800, 2200, 3200, 4200 }});
    else if (name == CHOIR_PAD)
        wt->generateVowelMorph();
    else
        wt->generateSaw();

    return wt;
}
