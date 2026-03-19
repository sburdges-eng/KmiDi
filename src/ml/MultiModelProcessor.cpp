#include "MultiModelProcessor.h"
#include "ml/ModelConfig.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>

namespace Kelly {
namespace ML {

namespace {

constexpr size_t kModelCount = static_cast<size_t>(ModelType::COUNT);

juce::String getPrimaryModelId(ModelType type) {
    switch (type) {
        case ModelType::EmotionRecognizer: return "emotionrecognizer";
        case ModelType::MelodyTransformer: return "melodytransformer";
        case ModelType::HarmonyPredictor:  return "harmonypredictor";
        case ModelType::DynamicsEngine:    return "dynamicsengine";
        case ModelType::GroovePredictor:   return "groovepredictor";
        case ModelType::AudioJEPA:         return "audiojepa";
        case ModelType::ChordJEPA:         return "chordjepa";
        case ModelType::LanguageModel:     return "languagemodel";
        case ModelType::COUNT:             break;
    }

    return {};
}

juce::StringArray getModelIds(ModelType type) {
    juce::StringArray ids{getPrimaryModelId(type)};

    switch (type) {
        case ModelType::AudioJEPA:
            ids.addIfNotAlreadyThere("audio_jepa");
            break;
        case ModelType::ChordJEPA:
            ids.addIfNotAlreadyThere("chord_jepa");
            break;
        case ModelType::LanguageModel:
            ids.addIfNotAlreadyThere("language_model");
            ids.addIfNotAlreadyThere("llama_onnx");
            break;
        default:
            break;
    }

    return ids;
}

bool prefersOnnx(ModelType type) {
    return type == ModelType::AudioJEPA ||
           type == ModelType::ChordJEPA ||
           type == ModelType::LanguageModel;
}

juce::String getStringProperty(const juce::DynamicObject& object, const juce::Identifier& key) {
    const auto value = object.getProperty(key);
    return value.isString() ? value.toString() : juce::String{};
}

juce::File resolveRelativeTo(const juce::File& baseDir, const juce::String& rawPath) {
    if (rawPath.isEmpty()) {
        return {};
    }

    juce::File file(rawPath);
    if (!juce::File::isAbsolutePath(rawPath)) {
        file = baseDir.getChildFile(rawPath);
    }
    return file;
}

float clampSigned(float value) {
    return std::clamp(value, -1.0f, 1.0f);
}

float clampUnit(float value) {
    return std::clamp(value, 0.0f, 1.0f);
}

float safeInput(const float* input, size_t index, size_t size) {
    if (input == nullptr || size == 0) {
        return 0.0f;
    }

    const float value = input[index % size];
    return std::isfinite(value) ? value : 0.0f;
}

} // namespace

// ============================================================================
// ModelWrapper Implementation
// ============================================================================

ModelWrapper::ModelWrapper(ModelType type)
    : type_(type), spec_(MODEL_SPECS[static_cast<size_t>(type)]) {
    output_.resize(spec_.outputSize, 0.0f);
}

juce::String ModelWrapper::getStatusSummary() const {
    juce::String backend = "fallback";
    if (!usingFallback_) {
#ifdef ENABLE_RTNEURAL
        if (rtModel_) {
            backend = "rtneural";
        } else
#endif
        if (onnxModel_ && onnxModel_->isLoaded()) {
            backend = "onnx";
        }
    }

    juce::String summary = backend + ": " + statusMessage_;
    if (modelSource_.isNotEmpty()) {
        summary += " [" + modelSource_ + "]";
    }
    return summary;
}

void ModelWrapper::setFallbackState(const juce::String& reason, const juce::File& file) {
    usingFallback_ = true;
    loaded_ = true;
    onnxModel_.reset();
#ifdef ENABLE_RTNEURAL
    rtModel_.reset();
#endif
    modelSource_ = file.getFullPathName();
    statusMessage_ = reason;
    juce::Logger::writeToLog(juce::String("Model ") + spec_.name + " using fallback: " + reason);
}

bool ModelWrapper::loadOnnxModel(const juce::File& file) {
    onnxModel_ = std::make_unique<midikompanion::ml::ONNXInference>();

    if (!onnxModel_->loadModel(file)) {
        const auto error = onnxModel_->getLastError();
        setFallbackState("ONNX load failed: " +
                         (error.isNotEmpty() ? error : juce::String("unknown error")), file);
        return false;
    }

    if (onnxModel_->getInputSize() != spec_.inputSize ||
        onnxModel_->getOutputSize() != spec_.outputSize) {
        setFallbackState("ONNX shape mismatch: expected " +
                         juce::String(spec_.inputSize) + "→" + juce::String(spec_.outputSize) +
                         ", got " + juce::String(onnxModel_->getInputSize()) + "→" +
                         juce::String(onnxModel_->getOutputSize()), file);
        return false;
    }

    usingFallback_ = false;
    loaded_ = true;
    modelSource_ = file.getFullPathName();
    statusMessage_ = "loaded";
    juce::Logger::writeToLog(juce::String("Loaded ONNX model: ") + spec_.name +
                             " (" + file.getFileName() + ")");
    return true;
}

bool ModelWrapper::loadConfiguredOnnxModel(const juce::File& file) {
    juce::var config;
    const auto parseResult = juce::JSON::parse(file.loadFileAsString(), config);
    if (parseResult.failed()) {
        return false;
    }

    const auto* object = config.getDynamicObject();
    if (object == nullptr || !object->hasProperty("model_path")) {
        return false;
    }

    const auto rawModelPath = object->getProperty("model_path").toString().trim();
    if (rawModelPath.isEmpty()) {
        setFallbackState("ONNX config missing model_path", file);
        return false;
    }

    const auto onnxPath = resolveRelativeTo(file.getParentDirectory(), rawModelPath);
    return loadOnnxModel(onnxPath);
}

bool ModelWrapper::loadWeights(const juce::File& file) {
    if (!file.existsAsFile()) {
        setFallbackState("model file not found", file);
        return false;
    }

    onnxModel_.reset();
#ifdef ENABLE_RTNEURAL
    rtModel_.reset();
#endif

    const auto extension = file.getFileExtension().toLowerCase();
    if (extension == ".onnx") {
        return loadOnnxModel(file);
    }

    if (extension == ".json" &&
        (file.getFileNameWithoutExtension().containsIgnoreCase("llama_onnx") ||
         file.loadFileAsString().contains("\"model_path\""))) {
        return loadConfiguredOnnxModel(file);
    }

#ifdef ENABLE_RTNEURAL
    try {
        std::string filePath = file.getFullPathName().toStdString();
        std::ifstream jsonStream(filePath, std::ifstream::binary);

        if (!jsonStream.is_open()) {
            setFallbackState("failed to open model file", file);
            return false;
        }

        auto model = RTNeural::json_parser::parseJson<float>(jsonStream);

        if (!model) {
            setFallbackState("failed to parse RTNeural model JSON", file);
            return false;
        }

        model->reset();
        rtModel_ = std::move(model);
        usingFallback_ = false;
        loaded_ = true;
        modelSource_ = file.getFullPathName();
        statusMessage_ = "loaded";
        juce::Logger::writeToLog(juce::String("Loaded RTNeural model: ") + spec_.name +
                                 " (" + juce::String(spec_.estimatedParams) + " params)");
        return true;
    }
    catch (const std::exception& e) {
        setFallbackState("RTNeural exception: " + juce::String(e.what()), file);
        return false;
    }
    catch (...) {
        setFallbackState("unknown RTNeural exception", file);
        return false;
    }
#else
    setFallbackState("RTNeural disabled", file);
    return false;
#endif
}

std::vector<float> ModelWrapper::forward(const float* input, size_t inputSize) {
    if (!enabled_) {
        std::fill(output_.begin(), output_.end(), 0.0f);
        return output_;
    }

    if (!loaded_) {
        return output_;
    }

    if (inputSize != spec_.inputSize) {
        juce::Logger::writeToLog(juce::String("ModelWrapper: Input size mismatch. Expected ") +
                                 juce::String(spec_.inputSize) + ", got " + juce::String(inputSize));
        std::fill(output_.begin(), output_.end(), 0.0f);
        return output_;
    }

    if (onnxModel_ && onnxModel_->isLoaded()) {
        if (onnxModel_->infer(input, output_.data())) {
            sanitizeOutput();
            return output_;
        }

        juce::Logger::writeToLog(juce::String("ONNX inference failed for ") + spec_.name +
                                 ": " + onnxModel_->getLastError());
    }

#ifdef ENABLE_RTNEURAL
    if (rtModel_) {
        try {
            rtModel_->forward(input);
            const float* modelOutput = rtModel_->getOutputs();
            if (modelOutput) {
                size_t copySize = std::min(spec_.outputSize, output_.size());
                std::memcpy(output_.data(), modelOutput, copySize * sizeof(float));
                sanitizeOutput();
            } else {
                computeFallback(input);
            }
            return output_;
        }
        catch (const std::exception& e) {
            juce::Logger::writeToLog(juce::String("Exception during inference: ") +
                                     juce::String(e.what()));
            computeFallback(input);
            return output_;
        }
    }
#endif

    computeFallback(input);
    return output_;
}

void ModelWrapper::computeFallback(const float* input) {
    std::fill(output_.begin(), output_.end(), 0.0f);

    switch (type_) {
        case ModelType::EmotionRecognizer:
            for (size_t i = 0; i < 32 && i < output_.size(); ++i) {
                output_[i] = clampSigned(std::tanh(safeInput(input, i, spec_.inputSize) * 0.3f));
                if (i + 32 < output_.size()) {
                    output_[i + 32] = clampSigned(std::tanh(
                        safeInput(input, i + 64, spec_.inputSize) * 0.5f));
                }
            }
            break;

        case ModelType::MelodyTransformer:
            for (size_t i = 0; i < output_.size(); ++i) {
                float note = static_cast<float>(i % 12) / 12.0f;
                float emotionInfluence = safeInput(input, i, spec_.inputSize);
                const float raw = std::clamp(note * 2.0f - 1.0f + emotionInfluence * 0.5f,
                                             -12.0f, 12.0f);
                output_[i] = clampUnit(1.0f / (1.0f + std::exp(-raw)));
            }
            break;

        case ModelType::HarmonyPredictor:
            for (size_t i = 0; i < output_.size(); ++i) {
                output_[i] = clampSigned(std::tanh(safeInput(input, i, spec_.inputSize) * 0.4f));
            }
            break;

        case ModelType::DynamicsEngine:
            for (size_t i = 0; i < output_.size(); ++i) {
                output_[i] = clampUnit(0.5f + safeInput(input, i, spec_.inputSize) * 0.3f);
            }
            break;

        case ModelType::GroovePredictor:
            for (size_t i = 0; i < output_.size(); ++i) {
                output_[i] = clampSigned(std::tanh(safeInput(input, i, spec_.inputSize) * 0.6f));
            }
            break;

        case ModelType::AudioJEPA: {
            float mean = 0.0f;
            float energy = 0.0f;
            float slope = 0.0f;
            for (size_t i = 0; i < spec_.inputSize; ++i) {
                const float current = safeInput(input, i, spec_.inputSize);
                mean += current;
                energy += current * current;
                if (i > 0) {
                    slope += std::abs(current - safeInput(input, i - 1, spec_.inputSize));
                }
            }

            mean /= static_cast<float>(spec_.inputSize);
            energy /= static_cast<float>(spec_.inputSize);
            slope /= static_cast<float>(std::max<size_t>(1, spec_.inputSize - 1));

            for (size_t i = 0; i < output_.size(); ++i) {
                const float left = safeInput(input, i * 2, spec_.inputSize);
                const float right = safeInput(input, i * 5 + 3, spec_.inputSize);
                const float mixed = 0.45f * left + 0.25f * (right - left) +
                                    0.20f * mean + 0.10f * (energy - slope);
                output_[i] = clampSigned(std::tanh(mixed));
            }
            break;
        }

        case ModelType::ChordJEPA: {
            float harmonicMean = 0.0f;
            float harmonicSpread = 0.0f;
            for (size_t i = 0; i < spec_.inputSize; ++i) {
                harmonicMean += safeInput(input, i, spec_.inputSize);
            }
            harmonicMean /= static_cast<float>(spec_.inputSize);

            for (size_t i = 0; i < spec_.inputSize; ++i) {
                harmonicSpread += std::abs(safeInput(input, i, spec_.inputSize) - harmonicMean);
            }
            harmonicSpread /= static_cast<float>(spec_.inputSize);

            for (size_t i = 0; i < output_.size(); ++i) {
                const float root = safeInput(input, i, spec_.inputSize);
                const float color = safeInput(input, i * 3 + 7, spec_.inputSize);
                const float mixed = 0.50f * root + 0.20f * color +
                                    0.20f * harmonicMean - 0.10f * harmonicSpread;
                output_[i] = clampSigned(std::tanh(mixed));
            }
            break;
        }

        case ModelType::LanguageModel: {
            constexpr size_t audioSpan = 128;
            constexpr size_t chordSpan = 64;
            constexpr size_t emotionSpan = 64;

            float audioMean = 0.0f;
            float chordMean = 0.0f;
            float emotionMean = 0.0f;
            float latentEnergy = 0.0f;

            for (size_t i = 0; i < audioSpan; ++i) {
                const float value = safeInput(input, i, spec_.inputSize);
                audioMean += value;
                latentEnergy += value * value;
            }
            for (size_t i = 0; i < chordSpan; ++i) {
                chordMean += safeInput(input, audioSpan + i, spec_.inputSize);
            }
            for (size_t i = 0; i < emotionSpan; ++i) {
                emotionMean += safeInput(input, audioSpan + chordSpan + i, spec_.inputSize);
            }

            audioMean /= static_cast<float>(audioSpan);
            chordMean /= static_cast<float>(chordSpan);
            emotionMean /= static_cast<float>(emotionSpan);
            latentEnergy /= static_cast<float>(audioSpan);

            for (size_t i = 0; i < output_.size(); ++i) {
                const float local = safeInput(input, i * 7 + 11, spec_.inputSize);
                const float paired = safeInput(input, i * 5 + 29, spec_.inputSize);
                const float raw = std::clamp(
                    0.45f * local +
                    0.20f * paired +
                    0.15f * audioMean +
                    0.10f * chordMean +
                    0.10f * emotionMean -
                    0.05f * latentEnergy,
                    -12.0f, 12.0f);
                output_[i] = clampUnit(1.0f / (1.0f + std::exp(-raw)));
            }
            break;
        }

        default:
            break;
    }

    sanitizeOutput();
}

void ModelWrapper::sanitizeOutput() {
    for (auto& value : output_) {
        if (!std::isfinite(value)) {
            value = 0.0f;
            continue;
        }

        switch (type_) {
            case ModelType::MelodyTransformer:
            case ModelType::DynamicsEngine:
            case ModelType::LanguageModel:
                value = clampUnit(value);
                break;
            default:
                value = clampSigned(value);
                break;
        }
    }
}

// ============================================================================
// MultiModelProcessor Implementation
// ============================================================================

MultiModelProcessor::MultiModelProcessor() {
    for (size_t i = 0; i < kModelCount; ++i) {
        models_[i] = std::make_unique<ModelWrapper>(static_cast<ModelType>(i));
    }
}

juce::File MultiModelProcessor::resolveModelFile(const juce::File& modelsDir, ModelType type) const {
    const auto registryFile = modelsDir.getChildFile("registry.json");
    juce::var registry;

    if (registryFile.existsAsFile() &&
        juce::JSON::parse(registryFile.loadFileAsString(), registry).wasOk()) {
        if (const auto* root = registry.getDynamicObject()) {
            const auto modelsVar = root->getProperty("models");
            if (const auto* models = modelsVar.getArray()) {
                const auto ids = getModelIds(type);
                for (const auto& modelVar : *models) {
                    const auto* entry = modelVar.getDynamicObject();
                    if (entry == nullptr) {
                        continue;
                    }

                    const auto entryId =
                        getStringProperty(*entry, juce::Identifier("id")).toLowerCase();
                    if (!ids.contains(entryId)) {
                        continue;
                    }

                    const auto filePath =
                        resolveRelativeTo(modelsDir, getStringProperty(*entry, juce::Identifier("file")));
                    juce::String onnxPathString = getStringProperty(*entry, juce::Identifier("onnx_path"));
                    if (onnxPathString.isEmpty()) {
                        const auto exportsVar = entry->getProperty("exports");
                        if (const auto* exports = exportsVar.getDynamicObject()) {
                            onnxPathString = getStringProperty(*exports, juce::Identifier("onnx_path"));
                        }
                    }
                    const auto onnxPath = resolveRelativeTo(modelsDir, onnxPathString);

                    const auto first = prefersOnnx(type) ? onnxPath : filePath;
                    const auto second = prefersOnnx(type) ? filePath : onnxPath;
                    if (first.existsAsFile()) {
                        return first;
                    }
                    if (second.existsAsFile()) {
                        return second;
                    }
                    if (first.exists()) {
                        return first;
                    }
                    if (second.exists()) {
                        return second;
                    }
                    if (first.getFullPathName().isNotEmpty()) {
                        return first;
                    }
                    if (second.getFullPathName().isNotEmpty()) {
                        return second;
                    }
                }
            }
        }
    }

    const auto ids = getModelIds(type);
    juce::StringArray orderedExtensions;
    if (prefersOnnx(type)) {
        orderedExtensions.add(".onnx");
        orderedExtensions.add(".json");
    } else {
        orderedExtensions.add(".json");
        orderedExtensions.add(".onnx");
    }

    for (const auto& extension : orderedExtensions) {
        for (const auto& id : ids) {
            const auto candidate = modelsDir.getChildFile(id + extension);
            if (candidate.existsAsFile()) {
                return candidate;
            }
        }
    }

    const auto fallbackExtension = prefersOnnx(type) ? ".onnx" : ".json";
    return modelsDir.getChildFile(ids[0] + fallbackExtension);
}

bool MultiModelProcessor::loadRegistry(juce::File modelsDir) {
    bool anyLoaded = false;

    for (size_t i = 0; i < models_.size(); ++i) {
        const auto type = static_cast<ModelType>(i);
        const auto modelFile = resolveModelFile(modelsDir, type);
        if (models_[i]->loadWeights(modelFile)) {
            anyLoaded = true;
        }
    }

    return anyLoaded;
}

bool MultiModelProcessor::initialize(const juce::File& modelsDir) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!modelsDir.isDirectory()) {
        juce::Logger::writeToLog("Models directory not found: " + modelsDir.getFullPathName());
        juce::Logger::writeToLog("Will use fallback heuristics for all models");
    }

    const bool anyLoaded = loadRegistry(modelsDir);

    initialized_ = true;
    logStats();
    if (!anyLoaded) {
        juce::Logger::writeToLog(
            "MultiModelProcessor initialized with deterministic fallback heuristics only");
    }

    return initialized_;
}

void MultiModelProcessor::setModelEnabled(ModelType type, bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    models_[static_cast<size_t>(type)]->setEnabled(enabled);
}

bool MultiModelProcessor::isModelEnabled(ModelType type) const {
    return models_[static_cast<size_t>(type)]->isEnabled();
}

std::vector<float> MultiModelProcessor::infer(ModelType type, const std::vector<float>& input) {
    std::lock_guard<std::mutex> lock(mutex_);
    return models_[static_cast<size_t>(type)]->forward(input.data(), input.size());
}

InferenceResult MultiModelProcessor::runFullPipeline(const std::array<float, 128>& audioFeatures) {
    std::lock_guard<std::mutex> lock(mutex_);

    InferenceResult result;

    // 1. EmotionRecognizer: audio → emotion
    if (models_[0]->isEnabled()) {
        result.usedFallback[0] = models_[0]->isUsingFallback();
        auto emotion = models_[0]->forward(audioFeatures.data(), 128);
        std::copy_n(emotion.begin(), 64, result.emotionEmbedding.begin());
    }

    // 2. MelodyTransformer: emotion → melody
    if (models_[1]->isEnabled()) {
        result.usedFallback[1] = models_[1]->isUsingFallback();
        auto melody = models_[1]->forward(result.emotionEmbedding.data(), 64);
        std::copy_n(melody.begin(), 128, result.melodyProbabilities.begin());
    }

    // 3. HarmonyPredictor: context (emotion + audio) → harmony
    if (models_[2]->isEnabled()) {
        result.usedFallback[2] = models_[2]->isUsingFallback();
        std::array<float, 128> context{};
        std::copy_n(result.emotionEmbedding.begin(), 64, context.begin());
        std::copy_n(audioFeatures.begin(), 64, context.begin() + 64);
        auto harmony = models_[2]->forward(context.data(), 128);
        std::copy_n(harmony.begin(), 64, result.harmonyPrediction.begin());
    }

    // 4. DynamicsEngine: compact emotion → dynamics
    if (models_[3]->isEnabled()) {
        result.usedFallback[3] = models_[3]->isUsingFallback();
        std::array<float, 32> compact{};
        std::copy_n(result.emotionEmbedding.begin(), 32, compact.begin());
        auto dynamics = models_[3]->forward(compact.data(), 32);
        std::copy_n(dynamics.begin(), 16, result.dynamicsOutput.begin());
    }

    // 5. GroovePredictor: emotion → groove
    if (models_[4]->isEnabled()) {
        result.usedFallback[4] = models_[4]->isUsingFallback();
        auto groove = models_[4]->forward(result.emotionEmbedding.data(), 64);
        std::copy_n(groove.begin(), 32, result.grooveParameters.begin());
    }

    // 6. AudioJEPA: audio features → latent
    if (models_[5]->isEnabled()) {
        result.usedFallback[5] = models_[5]->isUsingFallback();
        // Expand 128 mel features to 256 for JEPA if needed, or use as is
        std::array<float, 256> expanded{};
        std::copy_n(audioFeatures.begin(), 128, expanded.begin());
        auto latent = models_[5]->forward(expanded.data(), 256);
        std::copy_n(latent.begin(), 128, result.audioJepaLatent.begin());
    }

    // 7. ChordJEPA: harmony → latent
    if (models_[6]->isEnabled()) {
        result.usedFallback[6] = models_[6]->isUsingFallback();
        std::array<float, 128> harmonyContext{};
        std::copy_n(result.harmonyPrediction.begin(), 64, harmonyContext.begin());
        auto latent = models_[6]->forward(harmonyContext.data(), 128);
        std::copy_n(latent.begin(), 64, result.chordJepaLatent.begin());
    }

    // 8. LanguageModel: JEPA latents + Emotion → Decisions
    if (models_[7]->isEnabled()) {
        result.usedFallback[7] = models_[7]->isUsingFallback();
        std::array<float, 512> llmInput{};
        std::copy_n(result.audioJepaLatent.begin(), 128, llmInput.begin());
        std::copy_n(result.chordJepaLatent.begin(), 64, llmInput.begin() + 128);
        std::copy_n(result.emotionEmbedding.begin(), 64, llmInput.begin() + 192);
        auto decision = models_[7]->forward(llmInput.data(), 512);
        std::copy_n(decision.begin(), 128, result.llmDecision.begin());
    }

    result.valid = true;
    return result;
}

size_t MultiModelProcessor::getTotalParams() const {
    size_t total = 0;
    for (const auto& spec : MODEL_SPECS) {
        total += spec.estimatedParams;
    }
    return total;
}

size_t MultiModelProcessor::getTotalMemoryKB() const {
    return getTotalParams() * sizeof(float) / 1024;
}

void MultiModelProcessor::logStats() const {
    juce::Logger::writeToLog("============================================================");
    juce::Logger::writeToLog("MultiModelProcessor initialized:");
    juce::Logger::writeToLog("  Total params: " + juce::String(getTotalParams()));
    juce::Logger::writeToLog("  Total memory: " + juce::String(getTotalMemoryKB()) + " KB");
    juce::Logger::writeToLog("  Estimated inference: <10ms");
    for (size_t i = 0; i < models_.size(); ++i) {
        juce::Logger::writeToLog("  " + getPrimaryModelId(static_cast<ModelType>(i)) +
                                 ": " + models_[i]->getStatusSummary());
    }
    juce::Logger::writeToLog("============================================================");
}

// ============================================================================
// AsyncMLPipeline Implementation
// ============================================================================

class InferenceThread : public juce::Thread {
public:
    explicit InferenceThread(AsyncMLPipeline& pipeline)
        : juce::Thread("Kelly ML Inference"), pipeline_(pipeline) {}

    void run() override {
        pipeline_.run();
    }

private:
    AsyncMLPipeline& pipeline_;
};

AsyncMLPipeline::AsyncMLPipeline(MultiModelProcessor& processor)
    : processor_(processor) {}

AsyncMLPipeline::~AsyncMLPipeline() {
    stop();
}

void AsyncMLPipeline::start() {
    if (running_) return;

    running_ = true;
    thread_ = std::make_unique<InferenceThread>(*this);
    static_cast<InferenceThread*>(thread_.get())->startThread();
}

void AsyncMLPipeline::stop() {
    running_ = false;

    if (thread_) {
        static_cast<InferenceThread*>(thread_.get())->stopThread(1000);
        thread_.reset();
    }
}

uint64_t AsyncMLPipeline::submitFeatures(const std::array<float, 128>& features) {
    // Reject new submissions if one is in-flight or an unread result exists.
    if (hasRequest_.load(std::memory_order_acquire) ||
        hasResult_.load(std::memory_order_acquire)) {
        return 0; // busy, reject submission
    }

    const uint64_t requestId = nextRequestId_.fetch_add(1, std::memory_order_acq_rel);

    // Clear any previous result so consumers wait for the new submission
    hasResult_.store(false, std::memory_order_release);
    pendingRequestId_.store(requestId, std::memory_order_release);
    pendingFeatures_ = features;
    hasRequest_.store(true, std::memory_order_release);
    return requestId;
}

bool AsyncMLPipeline::hasResult() const {
    return hasResult_.load(std::memory_order_acquire);
}

bool AsyncMLPipeline::hasResult(uint64_t requestId) const {
    return hasResult_.load(std::memory_order_acquire) &&
           latestResultId_.load(std::memory_order_acquire) == requestId;
}

InferenceResult AsyncMLPipeline::getResult() {
    hasResult_.store(false, std::memory_order_release);
    return latestResult_;
}

InferenceResult AsyncMLPipeline::getResult(uint64_t requestId) {
    if (!hasResult(requestId)) {
        return InferenceResult{};
    }

    hasResult_.store(false, std::memory_order_release);
    return latestResult_;
}

void AsyncMLPipeline::run() {
    while (running_) {
        if (hasRequest_.load(std::memory_order_acquire)) {
            hasRequest_.store(false, std::memory_order_release);

            // Run inference on background thread
            latestResult_ = processor_.runFullPipeline(pendingFeatures_);
            latestResultId_.store(pendingRequestId_.load(std::memory_order_acquire),
                                  std::memory_order_release);

            hasResult_.store(true, std::memory_order_release);
        } else {
            juce::Thread::sleep(1);
        }
    }
}

} // namespace ML
} // namespace Kelly
