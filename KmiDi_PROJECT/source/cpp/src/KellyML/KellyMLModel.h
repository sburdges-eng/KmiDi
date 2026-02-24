#pragma once

#include <string>
#include <cstddef>

// Minimal RTNeural forward declaration to avoid pulling heavy headers here.
#ifdef ENABLE_RTNEURAL
namespace RTNeural {
    template <typename T>
    class Model;
}
#endif

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

    bool isLoaded() const noexcept {
#ifdef ENABLE_RTNEURAL
        return model_ != nullptr && enabled_;
#else
        return false; // RTNeural not available
#endif
    }
    void setEnabled(bool enabled) noexcept { enabled_ = enabled; }
    bool isEnabled() const noexcept { return enabled_; }

    // Processes input -> output. Returns false if not loaded/enabled.
    bool process(const float* input, float* output) noexcept;

    std::size_t inputSize() const noexcept { return inputSize_; }
    std::size_t outputSize() const noexcept { return outputSize_; }

private:
#ifdef ENABLE_RTNEURAL
    std::unique_ptr<RTNeural::Model<float>> model_;
#endif
    std::size_t inputSize_ {0};
    std::size_t outputSize_ {0};
    bool enabled_ {true};
};

} // namespace kelly::ml
