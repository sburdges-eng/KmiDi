#ifndef KMIDI_JEPA_RUNNER_H
#define KMIDI_JEPA_RUNNER_H

#include <string>
#include <vector>

namespace KmiDi {

class JepaRunner {
public:
    JepaRunner();
    ~JepaRunner();
    JepaRunner(const JepaRunner&) = delete;
    JepaRunner& operator=(const JepaRunner&) = delete;
    JepaRunner(JepaRunner&&) = delete;
    JepaRunner& operator=(JepaRunner&&) = delete;
    bool load(const std::string& onnx_path);
    std::vector<float> infer(const std::vector<float>& input);
    bool isLoaded() const { return loaded_; }

private:
    bool loaded_ = false;
    void* session_ = nullptr;
    void* env_ = nullptr;
    size_t input_size_ = 0;
    size_t output_size_ = 0;
};

}  // namespace KmiDi

#endif  // KMIDI_JEPA_RUNNER_H
