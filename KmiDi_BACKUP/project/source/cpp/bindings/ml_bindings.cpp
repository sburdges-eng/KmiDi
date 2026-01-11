#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cstring>

#include "penta/ml/MLInterface.h"

namespace py = pybind11;
using namespace penta::ml;

void bind_ml(py::module_& m) {
    py::enum_<ModelType>(m, "ModelType")
        .value("ChordPredictor", ModelType::ChordPredictor)
        .value("GrooveTransfer", ModelType::GrooveTransfer)
        .value("KeyDetector", ModelType::KeyDetector)
        .value("IntentMapper", ModelType::IntentMapper)
        .value("EmotionRecognizer", ModelType::EmotionRecognizer)
        .value("MelodyTransformer", ModelType::MelodyTransformer)
        .value("HarmonyPredictor", ModelType::HarmonyPredictor)
        .value("DynamicsEngine", ModelType::DynamicsEngine)
        .value("GroovePredictor", ModelType::GroovePredictor)
        .value("Custom", ModelType::Custom)
        .export_values();

    py::class_<MLConfig>(m, "MLConfig")
        .def(py::init<>())
        .def_readwrite("model_directory", &MLConfig::model_directory)
        .def_readwrite("max_concurrent_requests", &MLConfig::max_concurrent_requests)
        .def_readwrite("use_gpu", &MLConfig::use_gpu)
        .def_readwrite("use_coreml", &MLConfig::use_coreml)
        .def_readwrite("inference_thread_priority", &MLConfig::inference_thread_priority)
        .def_readwrite("timeout_ms", &MLConfig::timeout_ms);

    py::class_<MLInterface::Stats>(m, "MLStats")
        .def_readonly("total_requests", &MLInterface::Stats::total_requests)
        .def_readonly("completed_requests", &MLInterface::Stats::completed_requests)
        .def_readonly("failed_requests", &MLInterface::Stats::failed_requests)
        .def_readonly("queue_overflows", &MLInterface::Stats::queue_overflows)
        .def_readonly("avg_latency_ms", &MLInterface::Stats::avg_latency_ms)
        .def_readonly("max_latency_ms", &MLInterface::Stats::max_latency_ms);

    py::class_<MLInterface>(m, "MLInterface")
        .def(py::init<const MLConfig&>(), py::arg("config"))
        .def("start", &MLInterface::start, "Start inference thread")
        .def("stop", &MLInterface::stop, "Stop inference thread")
        .def("is_running", &MLInterface::isRunning, "Check if inference is running")
        .def("load_model", &MLInterface::loadModel, py::arg("type"), py::arg("path"),
             "Load a single model file")
        .def("load_registry", &MLInterface::loadRegistry, py::arg("registry_path"),
             "Load all models from registry.json")
        .def("is_model_loaded", &MLInterface::isModelLoaded, py::arg("type"),
             "Check if a model is loaded")
        .def("get_stats", &MLInterface::getStats, "Get inference statistics")
        .def("next_request_id", &MLInterface::getNextRequestId,
             "Get next request id")
        .def("submit_features",
             [](MLInterface& self,
                ModelType type,
                py::array_t<float, py::array::c_style | py::array::forcecast> features,
                uint64_t timestamp) {
                 py::buffer_info info = features.request();
                 if (info.ndim != 1) {
                     throw std::runtime_error("features must be 1-D array");
                 }
                 const size_t len = std::min<size_t>(info.shape[0], 128);

                 InferenceRequest req{};
                 req.model_type = type;
                 req.input_size = len;
                 req.timestamp = timestamp;
                 req.request_id = self.getNextRequestId();
                 std::memcpy(req.input_data.data(), info.ptr, len * sizeof(float));

                 const bool queued = self.submitRequest(req);
                 return py::make_tuple(queued, req.request_id);
             },
             py::arg("model_type"),
             py::arg("features"),
             py::arg("timestamp") = 0,
             "Submit inference request with feature vector. Returns (queued, request_id).")
        .def("poll_result",
             [](MLInterface& self) -> py::object {
                 InferenceResult res{};
                 if (!self.pollResult(res)) {
                     return py::none();
                 }
                 py::dict out;
                 out["model_type"] = res.model_type;
                 out["request_id"] = res.request_id;
                 out["success"] = res.success;
                 out["confidence"] = res.confidence;
                 out["latency_ms"] = res.latency_ms;
                 out["output_size"] = res.output_size;
                 py::list data;
                 for (size_t i = 0; i < res.output_size && i < res.output_data.size(); ++i) {
                     data.append(res.output_data[i]);
                 }
                 out["output"] = std::move(data);
                 return out;
             },
             "Poll for next inference result. Returns dict or None if empty.");
}

