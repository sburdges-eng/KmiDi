/*
 * KmiDi JEPA ONNX Runtime wrapper.
 * Build with ONNX Runtime C++: add include path and link -lonnxruntime.
 * Without ONNX Runtime, stub implementation returns empty; plugin still compiles.
 */

#include "JepaRunner.h"
#include <cstring>
#include <stdexcept>

// Stub implementation: replace with ONNX Runtime C++ API when building with libonnxruntime.
// See: https://onnxruntime.ai/docs/get-started/with C++.html

namespace KmiDi {

struct JepaRunnerImpl {
    std::string model_path;
    size_t input_size = 0;
    size_t output_size = 0;
    bool session_created = false;
};

static JepaRunnerImpl* impl(JepaRunner* self) {
    return reinterpret_cast<JepaRunnerImpl*>(self->session_);
}

JepaRunner::JepaRunner() {
    session_ = new JepaRunnerImpl();
}

JepaRunner::~JepaRunner() {
    delete impl(this);
    session_ = nullptr;
}

bool JepaRunner::load(const std::string& onnx_path) {
    auto* p = impl(this);
    p->model_path = onnx_path;
    // TODO: Ort::Env env; Ort::Session session(env, onnx_path.c_str(), ...);
    // For stub: assume 1*1*128*512 input, variable output size
    p->input_size = 1 * 1 * 128 * 512;
    p->output_size = 256 * 128;  // example latent shape
    p->session_created = true;
    loaded_ = true;
    return true;
}

std::vector<float> JepaRunner::infer(const std::vector<float>& input) {
    if (!loaded_ || session_ == nullptr) {
        throw std::runtime_error("JepaRunner is not loaded");
    }
    auto* p = impl(this);
    if (!p->session_created) {
        throw std::runtime_error("JepaRunner session is not initialized");
    }
    if (p->input_size > 0 && input.size() != p->input_size) {
        throw std::runtime_error("JepaRunner input shape mismatch");
    }

    // Fail fast in stub mode so missing ONNX integration is explicit.
    throw std::runtime_error(
        "JepaRunner stub cannot execute inference. "
        "Build with ONNX Runtime support and implement session execution.");
}

}  // namespace KmiDi
