#include "KellyMLModel.h"

#ifdef ENABLE_RTNEURAL
#include <RTNeural/ModelT.h>
#include <RTNeural/ModelLoader.h>
#endif

namespace kelly::ml {

// Destructor must be defined here (not = default) to ensure RTNeural::Model<float>
// is fully defined when unique_ptr destructor is instantiated
KellyMLModel::~KellyMLModel() {
#ifdef ENABLE_RTNEURAL
    // unique_ptr will automatically destroy model_ if it exists
    // RTNeural headers are included above, so Model<float> is complete here
#endif
    // When RTNeural is not enabled, there's nothing to destroy
}

KellyMLModel::KellyMLModel(KellyMLModel&& other) noexcept
#ifdef ENABLE_RTNEURAL
    : model_(std::move(other.model_))
#endif
    , inputSize_(other.inputSize_)
    , outputSize_(other.outputSize_)
    , enabled_(other.enabled_) {}

KellyMLModel& KellyMLModel::operator=(KellyMLModel&& other) noexcept {
    if (this != &other) {
#ifdef ENABLE_RTNEURAL
        model_ = std::move(other.model_);
#endif
        inputSize_ = other.inputSize_;
        outputSize_ = other.outputSize_;
        enabled_ = other.enabled_;
    }
    return *this;
}

bool KellyMLModel::loadFromJson(const std::string& path, std::size_t inputSize, std::size_t outputSize) {
    inputSize_ = inputSize;
    outputSize_ = outputSize;

#ifdef ENABLE_RTNEURAL
    try {
        model_.reset(RTNeural::json_parser::parseJson<float>(path));
        if (!model_) {
            return false;
        }
        if (model_->getInputSize() != static_cast<int>(inputSize_) ||
            model_->getOutputSize() != static_cast<int>(outputSize_)) {
            model_.reset();
            return false;
        }
        return true;
    } catch (...) {
        model_.reset();
        return false;
    }
#else
    // RTNeural not available - return false
    return false;
#endif
}

bool KellyMLModel::process(const float* input, float* output) noexcept {
    if (!isLoaded() || input == nullptr || output == nullptr) {
        return false;
    }

#ifdef ENABLE_RTNEURAL
    // RTNeural models work in-place; copy input to a fixed stack buffer (no heap on audio thread).
    float buffer[256] = {};
    if (inputSize_ > 256) {
        return false; // defensive: unsupported size for stack buffer
    }
    for (std::size_t i = 0; i < inputSize_; ++i) buffer[i] = input[i];

    model_->forward(buffer);

    for (std::size_t i = 0; i < outputSize_; ++i) {
        output[i] = model_->getOutputs()[i];
    }
    return true;
#else
    // RTNeural not available - cannot process
    return false;
#endif
}

} // namespace kelly::ml
