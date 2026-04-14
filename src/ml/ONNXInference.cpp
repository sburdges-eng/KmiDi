#include "ml/ONNXInference.h"
#include <juce_core/juce_core.h>
#include <cstring>

#ifdef ENABLE_ONNX_RUNTIME
// Include ONNX Runtime headers
#include <onnxruntime_cxx_api.h>
using namespace Ort;

// Thin owning wrappers so unique_ptr can work with incomplete types in the header.
struct OrtEnvOwner     { Ort::Env       value; template<typename... A> OrtEnvOwner(A&&... a)     : value(std::forward<A>(a)...) {} };
struct OrtSessionOwner { Ort::Session   value; template<typename... A> OrtSessionOwner(A&&... a) : value(std::forward<A>(a)...) {} };
struct OrtMemInfoOwner { Ort::MemoryInfo value; template<typename... A> OrtMemInfoOwner(A&&... a): value(std::forward<A>(a)...) {} };
#endif

namespace midikompanion {
namespace ml {

// Static initialization
bool ONNXInference::onnxInitialized_ = false;
std::mutex ONNXInference::initMutex_;

ONNXInference::ONNXInference()
    : inputSize_(0)
    , outputSize_(0)
    , isLoaded_(false)
{
    clearError();
}

ONNXInference::~ONNXInference() {
    // unique_ptr members clean up automatically; explicit ordering: session before env.
#ifdef ENABLE_ONNX_RUNTIME
    memInfoOwner_.reset();
    sessionOwner_.reset();
    envOwner_.reset();
#endif
}

void ONNXInference::initializeONNX() {
    std::lock_guard<std::mutex> lock(initMutex_);

    if (onnxInitialized_) {
        return;
    }

#ifdef ENABLE_ONNX_RUNTIME
    try {
        // Initialize ONNX Runtime environment (global, initialized once)
        // Note: Ort::Env is thread-safe and can be initialized multiple times safely
        // We use a static flag to avoid redundant initialization
        onnxInitialized_ = true;
    } catch (const std::exception& e) {
        setError("Failed to initialize ONNX Runtime: " + juce::String(e.what()));
        onnxInitialized_ = false;
    }
#else
    // ONNX Runtime not available - stub mode
    onnxInitialized_ = true;  // Allow stub mode to work
#endif
}

bool ONNXInference::loadModel(const juce::File& modelPath) {
    return loadModel(modelPath.getFullPathName().toStdString());
}

bool ONNXInference::loadModel(const std::string& modelPath) {
    clearError();

    if (!juce::File(modelPath).existsAsFile()) {
        setError("Model file does not exist: " + juce::String(modelPath));
        return false;
    }

    initializeONNX();

#ifdef ENABLE_ONNX_RUNTIME
    try {
        std::lock_guard<std::mutex> lock(mutex_);

        using namespace Ort;

        // Create ONNX Runtime environment if not already created
        if (!envOwner_) {
            envOwner_ = std::make_unique<OrtEnvOwner>(ORT_LOGGING_LEVEL_WARNING, "MidiKompanion");
        }

        // Reset previous session (unique_ptr handles the delete)
        sessionOwner_.reset();

        // Create session options
        SessionOptions sessionOptions;

        // Optimize for inference speed
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

        // Set execution providers (prefer CPU for now, can add GPU later)
        // For Apple platforms, could use CoreMLExecutionProvider
        // For NVIDIA, could use CUDAExecutionProvider

        // Create session — if this throws, envOwner_ is already held safely
        sessionOwner_ = std::make_unique<OrtSessionOwner>(envOwner_->value, modelPath.c_str(), sessionOptions);

        Session* session = &sessionOwner_->value;

        // Get input/output information
        size_t numInputNodes = session->GetInputCount();
        size_t numOutputNodes = session->GetOutputCount();

        if (numInputNodes == 0 || numOutputNodes == 0) {
            setError("Invalid model: no input or output nodes");
            return false;
        }

        // session already declared above, reuse it
        // Get input shape
        AllocatorWithDefaultOptions allocator;
        auto inputName = session->GetInputNameAllocated(0, allocator);
        auto inputTypeInfo = session->GetInputTypeInfo(0);
        auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        auto inputShape = inputTensorInfo.GetShape();
        inputShape_ = inputShape;

        // Calculate input size (handle dynamic dimensions)
        inputSize_ = 1;
        for (auto dim : inputShape) {
            if (dim > 0) {
                inputSize_ *= static_cast<size_t>(dim);
            } else {
                // Dynamic dimension - use default based on model type
                // For EmotionRecognizer, default to 128
                inputSize_ *= 128;
                break;
            }
        }

        // Get output shape
        auto outputName = session->GetOutputNameAllocated(0, allocator);
        auto outputTypeInfo = session->GetOutputTypeInfo(0);
        auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
        auto outputShape = outputTensorInfo.GetShape();
        outputShape_ = outputShape;

        // Calculate output size
        outputSize_ = 1;
        for (auto dim : outputShape) {
            if (dim > 0) {
                outputSize_ *= static_cast<size_t>(dim);
            } else {
                // Dynamic dimension - use default
                // For EmotionRecognizer, default to 64
                outputSize_ *= 64;
                break;
            }
        }

        // Reset previous memory info, then create new one
        memInfoOwner_.reset();
        memInfoOwner_ = std::make_unique<OrtMemInfoOwner>(MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

        modelPath_ = juce::String(modelPath);
        isLoaded_ = true;

        return true;

    } catch (const std::exception& e) {
        setError("Failed to load ONNX model: " + juce::String(e.what()));
        isLoaded_ = false;
        return false;
    }
#else
    // Stub mode: Return success but don't actually load
    // This allows code to compile without ONNX Runtime
    setError("ONNX Runtime not enabled. Set ENABLE_ONNX_RUNTIME=ON in CMake.");
    inputSize_ = 128;  // Default stub sizes
    outputSize_ = 64;
    inputShape_ = {1, static_cast<int64_t>(inputSize_)};
    outputShape_ = {1, static_cast<int64_t>(outputSize_)};
    modelPath_ = juce::String(modelPath);
    isLoaded_ = false;  // Mark as not loaded in stub mode
    return false;
#endif
}

std::vector<float> ONNXInference::infer(const std::vector<float>& input) {
    if (!isLoaded_) {
        setError("Model not loaded");
        return {};  // Return empty vector - error already logged via setError
    }

    if (!validateInputSize(input.size())) {
        setError("Input size validation failed: expected " + std::to_string(inputSize_) +
                 ", got " + std::to_string(input.size()));
        return {};  // Return empty vector - error logged
    }

    std::vector<float> output(outputSize_);

    if (infer(input.data(), output.data())) {
        return output;
    }

    // Error already set by infer() method
    return {};  // Return empty vector on inference failure
}

bool ONNXInference::infer(const float* input, float* output) {
    if (!isLoaded_) {
        setError("Model not loaded");
        return false;
    }

#ifdef ENABLE_ONNX_RUNTIME
    try {
        std::lock_guard<std::mutex> lock(mutex_);

        if (!sessionOwner_ || !memInfoOwner_) {
            setError("ONNX session not initialized");
            return false;
        }

        using namespace Ort;

        Session* session = &sessionOwner_->value;
        MemoryInfo* memoryInfo = &memInfoOwner_->value;

        AllocatorWithDefaultOptions allocator;

        // Get input/output names
        auto inputName = session->GetInputNameAllocated(0, allocator);
        auto outputName = session->GetOutputNameAllocated(0, allocator);

        // Create input tensor
        std::vector<int64_t> inputShape = inputShape_;
        if (inputShape.empty()) {
            inputShape = {1, static_cast<int64_t>(inputSize_)};
        }
        size_t knownProduct = 1;
        int dynamicCount = 0;
        for (auto dim : inputShape) {
            if (dim > 0) {
                knownProduct *= static_cast<size_t>(dim);
            } else {
                ++dynamicCount;
            }
        }
        if (dynamicCount > 0) {
            size_t remaining = inputSize_;
            if (knownProduct > 0) {
                remaining = inputSize_ / knownProduct;
            }
            for (auto& dim : inputShape) {
                if (dim <= 0) {
                    dim = static_cast<int64_t>(remaining > 0 ? remaining : 1);
                    remaining = 1;
                }
            }
        }
        Value inputTensor = Value::CreateTensor<float>(
            *memoryInfo,
            const_cast<float*>(input),  // ONNX Runtime doesn't modify, but API requires non-const
            inputSize_,
            inputShape.data(),
            inputShape.size()
        );

        // Run inference
        const char* inputNamePtr = inputName.get();
        const char* outputNamePtr = outputName.get();
        auto outputTensors = session->Run(
            RunOptions{nullptr},
            &inputNamePtr, &inputTensor, 1,
            &outputNamePtr, 1
        );

        if (outputTensors.empty()) {
            setError("Inference returned no outputs");
            return false;
        }

        // Extract output data
        float* outputData = outputTensors.front().GetTensorMutableData<float>();
        size_t outputCount = outputTensors.front().GetTensorTypeAndShapeInfo().GetElementCount();

        if (outputCount != outputSize_) {
            setError("Output size mismatch: expected " + juce::String(outputSize_) +
                     ", got " + juce::String(outputCount));
            return false;
        }

        // Copy output
        std::memcpy(output, outputData, outputSize_ * sizeof(float));

        return true;

    } catch (const std::exception& e) {
        setError("Inference failed: " + juce::String(e.what()));
        return false;
    }
#else
    // Stub mode: Return random data for testing
    setError("ONNX Runtime not enabled");
    for (size_t i = 0; i < outputSize_; ++i) {
        output[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;  // Random -1 to 1
    }
    return false;  // Return false to indicate stub mode
#endif
}

bool ONNXInference::validateInputSize(size_t size) {
    if (size != inputSize_) {
        setError("Input size mismatch: expected " + juce::String(inputSize_) +
                 ", got " + juce::String(size));
        return false;
    }
    return true;
}

bool ONNXInference::validateOutputSize(size_t size) {
    if (size != outputSize_) {
        setError("Output size mismatch: expected " + juce::String(outputSize_) +
                 ", got " + juce::String(size));
        return false;
    }
    return true;
}

} // namespace ml
} // namespace midikompanion
