#pragma once

#include <memory>
#include <string>
#include <vector>
#include <cstddef>

// Minimal RTNeural forward declaration to avoid pulling heavy headers here.
namespace RTNeural {
    template <typename T>
    class Model;
}

namespace kelly::ml {

// Lightweight wrapper around a single RTNeural model.
class KellyMLModel {
public:
    KellyMLModel() = default;
    ~KellyMLModel();

    KellyMLModel(const KellyMLModel&) = delete;
    KellyMLModel& operator=(const KellyMLModel&) = delete;
    KellyMLModel(KellyMLModel&&) noexcept;
    KellyMLModel& operator=(KellyMLModel&&) noexcept;

    bool loadFromJson(const std::string& path, std::size_t inputSize, std::size_t outputSize);

    bool isLoaded() const noexcept { return model_ != nullptr && enabled_; }
    void setEnabled(bool enabled) noexcept { enabled_ = enabled; }
    bool isEnabled() const noexcept { return enabled_; }

    // Processes input -> output. Returns false if not loaded/enabled.
    bool process(const float* input, float* output) noexcept;

    std::size_t inputSize() const noexcept { return inputSize_; }
    std::size_t outputSize() const noexcept { return outputSize_; }

private:
    std::unique_ptr<RTNeural::Model<float>> model_;
    std::size_t inputSize_ {0};
    std::size_t outputSize_ {0};
    bool enabled_ {true};
};

} // namespace kelly::ml
