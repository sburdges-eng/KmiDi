#include "OrchestratorBridge.h"
#include "bridge/PythonBridgeBase.h"

// Python C API - only include if Python is available
#ifdef PYTHON_AVAILABLE
#include "bridge/PyGILGuard.h"
#include <Python.h>
#endif

#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>

namespace kelly {

#ifdef PYTHON_AVAILABLE
namespace {

struct PyObjectDecref {
  void operator()(PyObject *obj) const noexcept {
    if (obj) {
      Py_DECREF(obj);
    }
  }
};

using PyOwnedObject = std::unique_ptr<PyObject, PyObjectDecref>;

PyOwnedObject makePyOwned(PyObject *obj) { return PyOwnedObject(obj); }

} // namespace
#endif

OrchestratorBridge::OrchestratorBridge()
    : executePipelineFunc_(nullptr), executePipelineAsyncFunc_(nullptr),
      getStatusFunc_(nullptr), cancelExecutionFunc_(nullptr) {
  available_.store(initializePython());
}

OrchestratorBridge::~OrchestratorBridge() {
  // Drain any in-flight workers that may not yet be in asyncThreads_.
  // shutdownPython() joins what's been pushed; this guards the race where
  // a worker incremented activeWorkers_ but hasn't been emplaced yet.
  while (activeWorkers_.load() > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  shutdownPython();
}

bool OrchestratorBridge::initializePython() {
#ifdef PYTHON_AVAILABLE
  // Delegate Py_Initialize + PyEval_SaveThread to the shared, thread-safe
  // helper.  This eliminates the race where two bridges constructed on
  // different threads both observe !Py_IsInitialized() and both call
  // PyEval_SaveThread (second call is UB).
  if (!bridge::PythonBridgeBase::ensurePythonStarted()) {
    std::cerr << "OrchestratorBridge: Python init failed\n";
    return false;
  }

  bridge::PyGILGuard gil;

  auto module =
      makePyOwned(PyImport_ImportModule("music_brain.orchestrator.bridge_api"));
  if (!module) {
    PyErr_Print();
    std::cerr
        << "OrchestratorBridge: Failed to import orchestrator bridge_api module"
        << std::endl;
    return false;
  }

  executePipelineFunc_ =
      PyObject_GetAttrString(module.get(), "execute_pipeline_sync");
  executePipelineAsyncFunc_ =
      PyObject_GetAttrString(module.get(), "execute_pipeline_async");
  getStatusFunc_ = PyObject_GetAttrString(module.get(), "get_execution_status");
  cancelExecutionFunc_ =
      PyObject_GetAttrString(module.get(), "cancel_execution");

  if (!executePipelineFunc_) {
    std::cerr
        << "OrchestratorBridge: Failed to get execute_pipeline_sync function"
        << std::endl;
    return false;
  }

  return true;
#else
  // Python not available - bridge will use fallback
  std::cerr << "OrchestratorBridge: Python not available, orchestrator disabled"
            << std::endl;
  return false;
#endif
}

void OrchestratorBridge::shutdownPython() {
  // Join all async threads before tearing down Python state
  {
    std::lock_guard<std::mutex> lock(asyncThreadsMutex_);
    for (auto &worker : asyncThreads_) {
      if (worker.thread.joinable())
        worker.thread.join();
    }
    asyncThreads_.clear();
  }

#ifdef PYTHON_AVAILABLE
  if (Py_IsInitialized()) {
    bridge::PyGILGuard
        gil; // safe: interpreter is alive; must hold GIL for Py_DECREF
    if (executePipelineFunc_) {
      Py_DECREF(static_cast<PyObject *>(executePipelineFunc_));
      executePipelineFunc_ = nullptr;
    }
    if (executePipelineAsyncFunc_) {
      Py_DECREF(static_cast<PyObject *>(executePipelineAsyncFunc_));
      executePipelineAsyncFunc_ = nullptr;
    }
    if (getStatusFunc_) {
      Py_DECREF(static_cast<PyObject *>(getStatusFunc_));
      getStatusFunc_ = nullptr;
    }
    if (cancelExecutionFunc_) {
      Py_DECREF(static_cast<PyObject *>(cancelExecutionFunc_));
      cancelExecutionFunc_ = nullptr;
    }
  } else {
    // Interpreter already finalized — just null the pointers; Py_DECREF is UB
    // here.
    executePipelineFunc_ = nullptr;
    executePipelineAsyncFunc_ = nullptr;
    getStatusFunc_ = nullptr;
    cancelExecutionFunc_ = nullptr;
  }
#endif
  activeExecutions_.clear();
  engineCallbacks_.clear();
}

std::string
OrchestratorBridge::executePipeline(const std::string &pipelineName,
                                    const std::string &inputDataJson) {
  if (!available_.load()) {
    return R"({"success": false, "error": "Python orchestrator not available"})";
  }

#ifdef PYTHON_AVAILABLE
  PyObject *func = static_cast<PyObject *>(executePipelineFunc_);
  if (!func) {
    return R"({"success": false, "error": "Pipeline execution function not available"})";
  }

  bridge::PyGILGuard gil; // must hold GIL for all Py* calls

  auto args = makePyOwned(PyTuple_New(2));
  if (!args) {
    return R"({"success": false, "error": "Argument allocation failed"})";
  }
  auto pipelineNameStr =
      makePyOwned(PyUnicode_FromString(pipelineName.c_str()));
  auto inputDataStr = makePyOwned(PyUnicode_FromString(inputDataJson.c_str()));
  if (!pipelineNameStr || !inputDataStr) {
    return R"({"success": false, "error": "Argument encoding failed"})";
  }

  PyTuple_SetItem(args.get(), 0, pipelineNameStr.release());
  PyTuple_SetItem(args.get(), 1, inputDataStr.release());

  auto result = makePyOwned(PyObject_CallObject(func, args.get()));

  if (!result) {
    PyErr_Print();
    return R"({"success": false, "error": "Pipeline execution failed"})";
  }

  std::string resultStr;
  if (PyUnicode_Check(result.get())) {
    const char *cstr = PyUnicode_AsUTF8(result.get());
    resultStr =
        cstr ? cstr : R"({"success": false, "error": "Invalid result format"})";
  } else {
    resultStr = R"({"success": false, "error": "Result is not a string"})";
  }

  return resultStr;
#else
  return R"({"success": false, "error": "Python not available"})";
#endif
}

void OrchestratorBridge::executePipelineAsync(
    const std::string &pipelineName, const std::string &inputDataJson,
    std::function<void(const std::string &)> callback) {
  if (!available_.load() || !callback) {
    return;
  }

  // Execute in background thread (stored, joined in destructor)
  {
    auto done = std::make_shared<std::atomic<bool>>(false);
    std::lock_guard<std::mutex> lock(asyncThreadsMutex_);
    reapCompletedAsyncTasksLocked();
    ++activeWorkers_;
    asyncThreads_.push_back(AsyncWorker{
        std::thread([this, pipelineName, inputDataJson, callback, done]() {
          std::string result = executePipeline(pipelineName, inputDataJson);
          callback(result);
          done->store(true, std::memory_order_release);
          --activeWorkers_;
        }),
        std::move(done)});
  }
}

std::string
OrchestratorBridge::getExecutionStatus(const std::string &executionId) {
  if (!available_.load() || !getStatusFunc_) {
    return R"({"status": "unknown", "error": "Status check not available"})";
  }

#ifdef PYTHON_AVAILABLE
  bridge::PyGILGuard gil; // must hold GIL for all Py* calls
  PyObject *func = static_cast<PyObject *>(getStatusFunc_);
  auto args = makePyOwned(PyTuple_New(1));
  if (!args) {
    return R"({"status": "unknown", "error": "Status argument allocation failed"})";
  }
  auto executionIdObj = makePyOwned(PyUnicode_FromString(executionId.c_str()));
  if (!executionIdObj) {
    return R"({"status": "unknown", "error": "Status argument encoding failed"})";
  }
  PyTuple_SetItem(args.get(), 0, executionIdObj.release());

  auto result = makePyOwned(PyObject_CallObject(func, args.get()));

  if (!result) {
    PyErr_Clear();
    return R"({"status": "unknown", "error": "Status check failed"})";
  }

  std::string resultStr;
  if (PyUnicode_Check(result.get())) {
    const char *cstr = PyUnicode_AsUTF8(result.get());
    resultStr = cstr ? cstr : R"({"status": "unknown"})";
  } else {
    resultStr = R"({"status": "unknown"})";
  }

  return resultStr;
#else
  return R"({"status": "unknown"})";
#endif
}

bool OrchestratorBridge::cancelExecution(const std::string &executionId) {
  if (!available_.load() || !cancelExecutionFunc_) {
    return false;
  }

#ifdef PYTHON_AVAILABLE
  bridge::PyGILGuard gil; // must hold GIL for all Py* calls
  PyObject *func = static_cast<PyObject *>(cancelExecutionFunc_);
  auto args = makePyOwned(PyTuple_New(1));
  if (!args) {
    return false;
  }
  auto executionIdObj = makePyOwned(PyUnicode_FromString(executionId.c_str()));
  if (!executionIdObj) {
    return false;
  }
  PyTuple_SetItem(args.get(), 0, executionIdObj.release());

  auto result = makePyOwned(PyObject_CallObject(func, args.get()));

  if (!result) {
    PyErr_Clear();
    return false;
  }

  return PyObject_IsTrue(result.get());
#else
  return false;
#endif
}

void OrchestratorBridge::registerEngineCallback(
    const std::string &engineType,
    std::function<std::string(const std::string &, const std::string &)>
        callback) {
  engineCallbacks_[engineType] = callback;
}

std::string OrchestratorBridge::generateExecutionId() {
  // Simple UUID-like generation
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 15);

  std::stringstream ss;
  ss << std::hex;
  for (int i = 0; i < 8; ++i) {
    ss << dis(gen);
  }
  ss << "-";
  for (int i = 0; i < 4; ++i) {
    ss << dis(gen);
  }
  ss << "-";
  for (int i = 0; i < 4; ++i) {
    ss << dis(gen);
  }
  ss << "-";
  for (int i = 0; i < 4; ++i) {
    ss << dis(gen);
  }
  ss << "-";
  for (int i = 0; i < 12; ++i) {
    ss << dis(gen);
  }

  return ss.str();
}

void OrchestratorBridge::reapCompletedAsyncTasksLocked() {
  auto out = asyncThreads_.begin();
  for (auto it = asyncThreads_.begin(); it != asyncThreads_.end(); ++it) {
    if (it->done && it->done->load(std::memory_order_acquire)) {
      if (it->thread.joinable()) {
        it->thread.join();
      }
      continue;
    }

    if (out != it) {
      *out = std::move(*it);
    }
    ++out;
  }
  asyncThreads_.erase(out, asyncThreads_.end());
}

} // namespace kelly
