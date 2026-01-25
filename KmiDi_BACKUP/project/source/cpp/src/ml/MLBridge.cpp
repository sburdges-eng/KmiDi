#include "ml/MLBridge.h"
#include "common/KellyTypes.h"
#include "engine/KellyBrain.h"
#include "engine/IntentPipeline.h"
#include "ml/MultiModelProcessor.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <memory>

namespace kelly {

// =============================================================================
// EmotionEmbeddingMapper Implementation
// =============================================================================

EmotionEmbeddingMapper::EmotionEmbeddingMapper(std::shared_ptr<EmotionThesaurus> thesaurus)
    : thesaurus_(std::move(thesaurus)) {
    initializeDefaultWeights();
}

void EmotionEmbeddingMapper::initializeDefaultWeights() {
    // Initialize category detection weights
    // Each category has a 64-dim weight vector
    // These would ideally be learned from data
    
    // Joy: High valence (dims 0-15), moderate arousal (dims 16-31)
    for (int i = 0; i < 16; ++i) {
        categoryWeights_[0][i] = 0.8f;      // Valence
        categoryWeights_[0][i + 16] = 0.4f; // Arousal
    }
    
    // Sadness: Low valence, low arousal
    for (int i = 0; i < 16; ++i) {
        categoryWeights_[1][i] = -0.7f;
        categoryWeights_[1][i + 16] = -0.3f;
    }
    
    // Anger: Low valence, high arousal
    for (int i = 0; i < 16; ++i) {
        categoryWeights_[2][i] = -0.5f;
        categoryWeights_[2][i + 16] = 0.8f;
    }
    
    // Fear: Low valence, high arousal, low dominance
    for (int i = 0; i < 16; ++i) {
        categoryWeights_[3][i] = -0.6f;
        categoryWeights_[3][i + 16] = 0.7f;
        categoryWeights_[3][i + 32] = -0.5f;
    }
    
    // Surprise: Neutral valence, high arousal
    for (int i = 0; i < 16; ++i) {
        categoryWeights_[4][i] = 0.1f;
        categoryWeights_[4][i + 16] = 0.8f;
    }
    
    // Disgust: Low valence, moderate arousal
    for (int i = 0; i < 16; ++i) {
        categoryWeights_[5][i] = -0.6f;
        categoryWeights_[5][i + 16] = 0.4f;
    }
    
    // Trust: High valence, low arousal
    for (int i = 0; i < 16; ++i) {
        categoryWeights_[6][i] = 0.6f;
        categoryWeights_[6][i + 16] = -0.2f;
    }
    
    // Anticipation: Moderate valence, moderate arousal
    for (int i = 0; i < 16; ++i) {
        categoryWeights_[7][i] = 0.3f;
        categoryWeights_[7][i + 16] = 0.5f;
    }
}

EmotionEmbeddingMapper::EmbeddingInterpretation 
EmotionEmbeddingMapper::interpret(const std::array<float, 64>& embedding) {
    EmbeddingInterpretation result;
    
    // Extract dimensional values
    // Dims 0-15: Valence
    float valenceSum = 0.0f;
    for (int i = 0; i < 16; ++i) valenceSum += embedding[i];
    result.valence = std::tanh(valenceSum / 8.0f);
    
    // Dims 16-31: Arousal
    float arousalSum = 0.0f;
    for (int i = 16; i < 32; ++i) arousalSum += embedding[i];
    result.arousal = (std::tanh(arousalSum / 8.0f) + 1.0f) / 2.0f;
    
    // Dims 32-47: Dominance
    float dominanceSum = 0.0f;
    for (int i = 32; i < 48; ++i) dominanceSum += embedding[i];
    result.dominance = (std::tanh(dominanceSum / 8.0f) + 1.0f) / 2.0f;
    
    // Dims 48-63: Complexity/Specificity
    float complexitySum = 0.0f;
    for (int i = 48; i < 64; ++i) complexitySum += std::abs(embedding[i]);
    result.complexity = complexitySum / 16.0f;
    
    // Determine primary category by dot product with category weights
    float maxScore = -1e9f;
    result.primaryCategory = EmotionCategory::Joy;
    
    for (int cat = 0; cat < 8; ++cat) {
        float score = 0.0f;
        for (int i = 0; i < 64; ++i) {
            score += embedding[i] * categoryWeights_[cat][i];
        }
        if (score > maxScore) {
            maxScore = score;
            result.primaryCategory = static_cast<EmotionCategory>(cat);
        }
    }
    
    // Confidence based on how strong the max score is
    result.confidence = std::min(1.0f, std::max(0.0f, maxScore / 10.0f + 0.5f));
    
    return result;
}

EmotionNode EmotionEmbeddingMapper::embeddingToEmotion(const std::array<float, 64>& embedding) {
    auto interp = interpret(embedding);
    
    EmotionNode node;
    node.valence = interp.valence;
    node.arousal = interp.arousal;
    node.dominance = interp.dominance;
    node.categoryEnum = interp.primaryCategory;
    node.category = categoryToString(interp.primaryCategory);
    node.intensity = interp.complexity;
    
    // Try to find matching node in thesaurus
    if (thesaurus_) {
        auto candidates = thesaurus_->getCategory(interp.primaryCategory);
        
        float minDist = 1e9f;
        const EmotionNode* bestMatch = nullptr;
        
        for (const auto* candidate : candidates) {
            float dv = candidate->valence - node.valence;
            float da = candidate->arousal - node.arousal;
            float dd = candidate->dominance - node.dominance;
            float dist = std::sqrt(dv*dv + da*da + dd*dd);
            
            if (dist < minDist) {
                minDist = dist;
                bestMatch = candidate;
            }
        }
        
        if (bestMatch && minDist < 0.5f) {
            node = *bestMatch;
            node.intensity = interp.complexity;
        } else {
            // Use category name as fallback
            node.name = categoryToString(interp.primaryCategory);
        }
    }
    
    // Derive musical attributes
    node.musicalAttributes.tempoModifier = 0.8f + node.arousal * 0.4f;
    node.musicalAttributes.mode = (node.valence < 0) ? "minor" : "major";
    node.musicalAttributes.dynamics = 0.3f + node.arousal * 0.5f;
    node.musicalAttributes.dissonance = std::max(0.0f, -node.valence * 0.5f);
    
    // Suggest rule breaks based on embedding
    if (node.valence < -0.3f) {
        node.musicalAttributes.suggestedRuleBreaks.push_back(RuleBreakType::ModalMixture);
    }
    if (node.arousal > 0.7f) {
        node.musicalAttributes.suggestedRuleBreaks.push_back(RuleBreakType::DynamicContrast);
    }
    if (node.dominance < 0.3f) {
        node.musicalAttributes.suggestedRuleBreaks.push_back(RuleBreakType::UnresolvedTension);
    }
    
    return node;
}

std::array<float, 64> EmotionEmbeddingMapper::emotionToEmbedding(const EmotionNode& node) {
    std::array<float, 64> embedding{};
    
    // Encode valence in dims 0-15
    for (int i = 0; i < 16; ++i) {
        embedding[i] = node.valence * (1.0f - static_cast<float>(i) / 32.0f);
    }
    
    // Encode arousal in dims 16-31
    for (int i = 16; i < 32; ++i) {
        embedding[i] = (node.arousal * 2.0f - 1.0f) * (1.0f - static_cast<float>(i - 16) / 32.0f);
    }
    
    // Encode dominance in dims 32-47
    for (int i = 32; i < 48; ++i) {
        embedding[i] = (node.dominance * 2.0f - 1.0f) * (1.0f - static_cast<float>(i - 32) / 32.0f);
    }
    
    // Encode intensity/complexity in dims 48-63
    for (int i = 48; i < 64; ++i) {
        embedding[i] = node.intensity * categoryWeights_[static_cast<int>(node.categoryEnum)][i];
    }
    
    return embedding;
}

// =============================================================================
// MLIntentPipeline Implementation
// =============================================================================

MLIntentPipeline::MLIntentPipeline() {
    intentPipeline_ = std::make_unique<IntentPipeline>();
    mlProcessor_ = std::make_unique<ML::MultiModelProcessor>();
    mapper_ = std::make_unique<EmotionEmbeddingMapper>(
        std::make_shared<EmotionThesaurus>(intentPipeline_->thesaurus()));
    featureExtractor_ = std::make_unique<ML::FeatureExtractor>(sampleRate_, fftSize_);
}

bool MLIntentPipeline::initialize(const std::string& modelsPath, const std::string& dataPath) {
    // Initialize ML models
    // Note: Models directory should contain the 5 model JSON files
    // mlProcessor_->initialize(juce::File(modelsPath));  // If using JUCE
    
    // Initialize intent pipeline data
    // intentPipeline_->thesaurus().loadFromFile(dataPath + "/emotions.json");
    
    // Start async pipeline
    asyncPipeline_ = std::make_unique<ML::AsyncMLPipeline>(*mlProcessor_);
    asyncPipeline_->start();
    
    initialized_ = true;
    return true;
}

IntentResult MLIntentPipeline::processAudio(const float* audioData, size_t numSamples) {
    if (!mlEnabled_ || !initialized_) {
        // Fallback to text-based processing
        return intentPipeline_->processText("audio input");
    }
    
    // Extract features
    auto features = featureExtractor_->extractFeatures(audioData, numSamples);
    
    // Run full ML pipeline
    auto mlResult = mlProcessor_->runFullPipeline(features);
    
    // Convert embedding to emotion
    EmotionNode emotion = mapper_->embeddingToEmotion(mlResult.emotionEmbedding);
    
    // Create wound from detected emotion
    Wound wound;
    wound.description = "Detected from audio";
    wound.primaryEmotion = emotion;
    wound.urgency = emotion.intensity;
    
    // Process through intent pipeline
    IntentResult result = intentPipeline_->processWound(wound);
    
    // Enhance with ML outputs
    // Store ML results for later use in generation
    result.metadata["ml_melody_ready"] = "true";
    result.metadata["ml_harmony_ready"] = "true";
    result.metadata["ml_dynamics_ready"] = "true";
    result.metadata["ml_groove_ready"] = "true";
    
    return result;
}

void MLIntentPipeline::submitAudio(const float* audioData, size_t numSamples) {
    if (!asyncPipeline_ || !mlEnabled_) return;
    
    auto features = featureExtractor_->extractFeatures(audioData, numSamples);
    asyncPipeline_->submitFeatures(features);
}

bool MLIntentPipeline::hasResult() const {
    return asyncPipeline_ && asyncPipeline_->hasResult();
}

IntentResult MLIntentPipeline::getResult() {
    if (!asyncPipeline_ || !asyncPipeline_->hasResult()) {
        return IntentResult{};
    }
    
    auto mlResult = asyncPipeline_->getResult();
    EmotionNode emotion = mapper_->embeddingToEmotion(mlResult.emotionEmbedding);
    
    Wound wound;
    wound.primaryEmotion = emotion;
    wound.urgency = emotion.intensity;
    
    return intentPipeline_->processWound(wound);
}

IntentResult MLIntentPipeline::processHybrid(
    const float* audioData, size_t numSamples,
    const std::string& textDescription) {
    
    // Get audio-based result
    IntentResult audioResult = processAudio(audioData, numSamples);
    
    // Get text-based result
    IntentResult textResult = intentPipeline_->processText(textDescription);
    
    // Blend results (favor text for explicit intent, audio for implicit emotion)
    IntentResult blended = textResult;
    
    // Use audio emotion intensity
    blended.sourceWound.primaryEmotion.intensity = 
        (audioResult.sourceWound.primaryEmotion.intensity + 
         textResult.sourceWound.primaryEmotion.intensity) / 2.0f;
    
    // Combine rule breaks
    for (const auto& rb : audioResult.ruleBreaks) {
        bool found = false;
        for (const auto& existing : blended.ruleBreaks) {
            if (existing.type == rb.type) {
                found = true;
                break;
            }
        }
        if (!found) {
            blended.ruleBreaks.push_back(rb);
        }
    }
    
    return blended;
}

std::vector<MidiNote> MLIntentPipeline::generateMelodyFromProbabilities(
    const std::array<float, 128>& noteProbabilities,
    const IntentResult& intent,
    int bars) {
    
    std::vector<MidiNote> notes;
    
    int ticksPerBar = TICKS_PER_BEAT * intent.timeSignature.numerator;
    int baseNote = noteNameToMidi(intent.key + "4");
    
    // Find top notes by probability
    std::vector<std::pair<float, int>> sortedNotes;
    for (int i = 0; i < 128; ++i) {
        sortedNotes.push_back({noteProbabilities[i], i});
    }
    std::sort(sortedNotes.begin(), sortedNotes.end(), std::greater<>());
    
    // Generate notes based on probabilities
    for (int bar = 0; bar < bars; ++bar) {
        int notesPerBar = 4;  // Quarter notes
        
        for (int beat = 0; beat < notesPerBar; ++beat) {
            // Select note based on probability
            int noteIdx = sortedNotes[beat % 12].second;  // Use top 12 notes
            
            // Constrain to reasonable range
            while (noteIdx < baseNote - 12) noteIdx += 12;
            while (noteIdx > baseNote + 12) noteIdx -= 12;
            
            MidiNote note;
            note.pitch = noteIdx;
            note.startTick = bar * ticksPerBar + beat * TICKS_PER_BEAT;
            note.durationTicks = TICKS_PER_BEAT;
            note.velocity = static_cast<int>(intent.baseVelocity * 127);
            
            notes.push_back(note);
        }
    }
    
    return notes;
}

std::vector<Chord> MLIntentPipeline::generateHarmonyFromPrediction(
    const std::array<float, 64>& harmonyPrediction,
    const IntentResult& intent) {
    
    // Map 64-dim prediction to chord qualities
    // First 12 dims: root note probabilities
    // Next 12 dims: chord quality (maj, min, dim, aug, 7, maj7, etc.)
    // Remaining: extensions and alterations
    
    std::vector<Chord> chords;
    
    static const char* roots[] = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
    static const char* qualities[] = {"", "m", "dim", "aug", "7", "maj7", "m7", "dim7", "sus2", "sus4", "add9", "6"};
    
    // Find top 4 chord predictions
    for (int i = 0; i < 4; ++i) {
        int offset = i * 16;
        
        // Find best root
        int bestRoot = 0;
        float maxRootProb = 0.0f;
        for (int r = 0; r < 12 && offset + r < 64; ++r) {
            if (harmonyPrediction[offset + r] > maxRootProb) {
                maxRootProb = harmonyPrediction[offset + r];
                bestRoot = r;
            }
        }
        
        // Determine quality based on remaining dims
        int qualityIdx = 0;  // Default major
        if (i * 16 + 12 < 64 && harmonyPrediction[i * 16 + 12] > 0.5f) {
            qualityIdx = 1;  // Minor
        }
        
        Chord chord;
        chord.root = roots[bestRoot];
        chord.quality = qualities[qualityIdx];
        chord.symbol = std::string(roots[bestRoot]) + qualities[qualityIdx];
        
        chords.push_back(chord);
    }
    
    return chords;
}

void MLIntentPipeline::applyDynamics(
    std::vector<MidiNote>& notes,
    const std::array<float, 16>& dynamicsOutput) {
    
    // Dynamics output interpretation:
    // 0-3: Velocity curve shape
    // 4-7: Timing micro-adjustments
    // 8-11: Articulation (staccato/legato)
    // 12-15: Expression/modulation
    
    float baseVelocityMod = dynamicsOutput[0];
    float velocityRange = dynamicsOutput[1];
    float timingSwing = dynamicsOutput[4];
    float articulationMod = dynamicsOutput[8];
    
    for (size_t i = 0; i < notes.size(); ++i) {
        auto& note = notes[i];
        
        // Velocity modulation
        float positionInPhrase = static_cast<float>(i % 4) / 4.0f;
        float velocityCurve = baseVelocityMod + velocityRange * std::sin(positionInPhrase * 3.14159f);
        note.velocity = std::clamp(
            static_cast<int>(note.velocity * (0.7f + velocityCurve * 0.6f)),
            1, 127);
        
        // Timing adjustment
        if (i % 2 == 1) {  // Off-beats
            note.startTick += static_cast<int>(timingSwing * 20.0f);
        }
        
        // Articulation (duration)
        note.durationTicks = static_cast<int>(
            note.durationTicks * (0.5f + articulationMod * 0.8f));
    }
}

void MLIntentPipeline::applyGroove(
    std::vector<MidiNote>& notes,
    const std::array<float, 32>& grooveParams) {
    
    // Groove output interpretation:
    // 0-15: 16th-note timing offsets
    // 16-31: 16th-note velocity accents
    
    for (auto& note : notes) {
        // Find position in 16th-note grid
        int sixteenthPos = (note.startTick / (TICKS_PER_BEAT / 4)) % 16;
        
        // Apply timing offset
        float timingOffset = grooveParams[sixteenthPos];
        note.startTick += static_cast<int>(timingOffset * (TICKS_PER_BEAT / 8));
        
        // Apply velocity accent
        float velocityMod = grooveParams[16 + sixteenthPos];
        note.velocity = std::clamp(
            static_cast<int>(note.velocity * (0.8f + velocityMod * 0.4f)),
            1, 127);
    }
}

GeneratedMidi MLIntentPipeline::generateFromAudio(
    const float* audioData, size_t numSamples, int bars) {
    
    IntentResult intent = processAudio(audioData, numSamples);
    
    GeneratedMidi result;
    result.tempoBpm = intent.tempoBpm;
    result.bars = bars;
    result.key = intent.key;
    result.mode = intent.mode;
    
    if (mlEnabled_) {
        // Get ML results
        auto features = featureExtractor_->extractFeatures(audioData, numSamples);
        auto mlResult = mlProcessor_->runFullPipeline(features);
        
        // Generate melody from ML probabilities
        result.notes = generateMelodyFromProbabilities(
            mlResult.melodyProbabilities, intent, bars);
        
        // Generate harmony from ML prediction
        result.chords = generateHarmonyFromPrediction(
            mlResult.harmonyPrediction, intent);
        
        // Apply dynamics
        applyDynamics(result.notes, mlResult.dynamicsOutput);
        
        // Apply groove
        applyGroove(result.notes, mlResult.grooveParameters);
    } else {
        // Fallback: basic generation
        KellyBrain brain;
        result = brain.generateMidi(intent, bars);
    }
    
    return result;
}

void MLIntentPipeline::setModelEnabled(ML::ModelType type, bool enabled) {
    mlProcessor_->setModelEnabled(type, enabled);
}

// =============================================================================
// FeatureExtractor Implementation
// =============================================================================

namespace ML {

FeatureExtractor::FeatureExtractor(double sampleRate, size_t fftSize)
    : sampleRate_(sampleRate)
    , fftSize_(fftSize)
    , hopSize_(fftSize / 4) {
    
    circularBuffer_.resize(fftSize_, 0.0f);
    windowFunction_.resize(fftSize_);
    
    // Hann window
    for (size_t i = 0; i < fftSize_; ++i) {
        windowFunction_[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (fftSize_ - 1)));
    }
    
    computeMelFilterbank();
}

void FeatureExtractor::prepare(double sampleRate, int samplesPerBlock) {
    sampleRate_ = sampleRate;
    reset();
    computeMelFilterbank();
}

void FeatureExtractor::reset() {
    std::fill(circularBuffer_.begin(), circularBuffer_.end(), 0.0f);
    bufferWritePos_ = 0;
    samplesAccumulated_ = 0;
}

void FeatureExtractor::computeMelFilterbank() {
    melFilterbank_.clear();
    melFilterbank_.resize(128);  // 128 mel bins
    
    float minMel = 2595.0f * std::log10(1.0f + 20.0f / 700.0f);
    float maxMel = 2595.0f * std::log10(1.0f + (sampleRate_ / 2.0f) / 700.0f);
    
    std::vector<float> melPoints(130);
    for (size_t i = 0; i < melPoints.size(); ++i) {
        float mel = minMel + (maxMel - minMel) * i / (melPoints.size() - 1);
        melPoints[i] = 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
    }
    
    std::vector<size_t> fftBins(melPoints.size());
    for (size_t i = 0; i < melPoints.size(); ++i) {
        fftBins[i] = static_cast<size_t>((fftSize_ + 1) * melPoints[i] / sampleRate_);
    }
    
    for (size_t m = 0; m < 128; ++m) {
        melFilterbank_[m].resize(fftSize_ / 2 + 1, 0.0f);
        
        for (size_t k = fftBins[m]; k < fftBins[m + 1] && k < melFilterbank_[m].size(); ++k) {
            melFilterbank_[m][k] = static_cast<float>(k - fftBins[m]) / 
                                   static_cast<float>(fftBins[m + 1] - fftBins[m]);
        }
        
        for (size_t k = fftBins[m + 1]; k < fftBins[m + 2] && k < melFilterbank_[m].size(); ++k) {
            melFilterbank_[m][k] = static_cast<float>(fftBins[m + 2] - k) / 
                                   static_cast<float>(fftBins[m + 2] - fftBins[m + 1]);
        }
    }
}

void FeatureExtractor::applyWindow(float* data, size_t size) {
    for (size_t i = 0; i < size && i < windowFunction_.size(); ++i) {
        data[i] *= windowFunction_[i];
    }
}

std::array<float, 128> FeatureExtractor::extractFeatures(const float* audioData, size_t numSamples) {
    std::array<float, 128> features{};
    
    // Copy to working buffer
    std::vector<float> workBuffer(fftSize_, 0.0f);
    size_t copySize = std::min(numSamples, fftSize_);
    std::copy(audioData, audioData + copySize, workBuffer.begin());
    
    // Apply window
    applyWindow(workBuffer.data(), fftSize_);
    
    // Simple DFT for magnitude spectrum
    std::vector<float> magnitudes(fftSize_ / 2 + 1, 0.0f);
    for (size_t k = 0; k < magnitudes.size(); ++k) {
        float real = 0.0f, imag = 0.0f;
        for (size_t n = 0; n < fftSize_; ++n) {
            float angle = -2.0f * M_PI * k * n / fftSize_;
            real += workBuffer[n] * std::cos(angle);
            imag += workBuffer[n] * std::sin(angle);
        }
        magnitudes[k] = std::sqrt(real * real + imag * imag);
    }
    
    // Apply mel filterbank
    for (size_t m = 0; m < 128; ++m) {
        float sum = 0.0f;
        for (size_t k = 0; k < magnitudes.size() && k < melFilterbank_[m].size(); ++k) {
            sum += magnitudes[k] * melFilterbank_[m][k];
        }
        features[m] = std::log(sum + 1e-10f);
    }
    
    return features;
}

void FeatureExtractor::pushSamples(const float* audioData, size_t numSamples) {
    for (size_t i = 0; i < numSamples; ++i) {
        circularBuffer_[bufferWritePos_] = audioData[i];
        bufferWritePos_ = (bufferWritePos_ + 1) % circularBuffer_.size();
        samplesAccumulated_++;
    }
}

bool FeatureExtractor::hasFeaturesReady() const {
    return samplesAccumulated_ >= hopSize_;
}

std::array<float, 128> FeatureExtractor::popFeatures() {
    // Linearize circular buffer
    std::vector<float> linear(fftSize_);
    for (size_t i = 0; i < fftSize_; ++i) {
        linear[i] = circularBuffer_[(bufferWritePos_ + i) % circularBuffer_.size()];
    }
    
    samplesAccumulated_ = 0;
    
    return extractFeatures(linear.data(), fftSize_);
}

} // namespace ML

} // namespace kelly
