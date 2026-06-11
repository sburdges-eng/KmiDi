#include "bridge/ContextBridge.h"

#ifdef PYTHON_AVAILABLE
#include "bridge/PyGILGuard.h"
#include <Python.h>
#endif

#include <functional>
#include <sstream>

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

ContextBridge::ContextBridge()
    : bridge::PythonBridgeBase("ContextBridge"), cache_(CACHE_TTL_MS, 50) {}

ContextBridge::~ContextBridge() { shutdown(); }

bool ContextBridge::initialize() {
  if (!PythonBridgeBase::initializePython()) {
    return false;
  }

  module_ = importModule("music_brain.intelligence.context_bridge");
  if (!module_) {
    return false;
  }

  analyzeContextFunc_ = getFunction(module_, "analyze_context");
  getContextualParametersFunc_ =
      getFunction(module_, "get_contextual_parameters");
  updateContextFunc_ = getFunction(module_, "update_context");
  getSuggestionsFunc_ = getFunction(module_, "get_contextual_suggestions");

  if (!analyzeContextFunc_) {
    logError("Failed to get analyze_context function");
    return false;
  }

  setAvailable(true);
  return true;
}

void ContextBridge::shutdown() {
  analyzeContextFunc_ = nullptr;
  getContextualParametersFunc_ = nullptr;
  updateContextFunc_ = nullptr;
  getSuggestionsFunc_ = nullptr;
  module_ = nullptr;
  cache_.clear();
  PythonBridgeBase::shutdownPython();
  setAvailable(false);
}

std::string ContextBridge::analyzeContext(const std::string &stateJson) {
  if (!isAvailable()) {
    return R"({"emotion_category": "unknown", "complexity_level": "moderate"})";
  }

  // Check cache first
  std::string cacheKey = hashState(stateJson);
  std::string cached = cache_.get(cacheKey);
  if (!cached.empty()) {
    return cached;
  }

#ifdef PYTHON_AVAILABLE
  if (!analyzeContextFunc_) {
    return R"({"emotion_category": "unknown"})";
  }

  bridge::PyGILGuard gil; // must hold GIL for all Py* calls
  auto args = makePyOwned(PyTuple_New(1));
  if (!args) {
    return R"({"emotion_category": "unknown"})";
  }

  auto stateStr = makePyOwned(PyUnicode_FromString(stateJson.c_str()));
  if (!stateStr) {
    return R"({"emotion_category": "unknown"})";
  }
  PyTuple_SetItem(args.get(), 0, stateStr.release());

  auto result =
      makePyOwned(PyObject_CallObject(analyzeContextFunc_, args.get()));

  if (!result) {
    PyErr_Print();
    return R"({"emotion_category": "unknown"})";
  }

  std::string resultStr;
  if (PyUnicode_Check(result.get())) {
    const char *cstr = PyUnicode_AsUTF8(result.get());
    resultStr = cstr ? cstr : R"({"emotion_category": "unknown"})";
  } else {
    resultStr = R"({"emotion_category": "unknown"})";
  }

  // Cache the result
  cache_.put(cacheKey, resultStr);

  return resultStr;
#else
  return R"({"emotion_category": "unknown"})";
#endif
}

std::string
ContextBridge::getContextualParameters(const std::string &stateJson) {
  if (!isAvailable() || !getContextualParametersFunc_) {
    return "{}";
  }

#ifdef PYTHON_AVAILABLE
  bridge::PyGILGuard gil; // must hold GIL for all Py* calls
  auto args = makePyOwned(PyTuple_New(1));
  if (!args) {
    return "{}";
  }
  auto stateStr = makePyOwned(PyUnicode_FromString(stateJson.c_str()));
  if (!stateStr) {
    return "{}";
  }
  PyTuple_SetItem(args.get(), 0, stateStr.release());

  auto result = makePyOwned(
      PyObject_CallObject(getContextualParametersFunc_, args.get()));

  if (!result) {
    PyErr_Clear();
    return "{}";
  }

  std::string resultStr;
  if (PyUnicode_Check(result.get())) {
    const char *cstr = PyUnicode_AsUTF8(result.get());
    resultStr = cstr ? cstr : "{}";
  } else {
    resultStr = "{}";
  }

  return resultStr;
#else
  return "{}";
#endif
}

void ContextBridge::updateContext(const std::string &stateJson) {
  if (!isAvailable() || !updateContextFunc_) {
    return;
  }

#ifdef PYTHON_AVAILABLE
  bridge::PyGILGuard gil; // must hold GIL for all Py* calls
  auto args = makePyOwned(PyTuple_New(1));
  if (!args) {
    return;
  }
  auto stateStr = makePyOwned(PyUnicode_FromString(stateJson.c_str()));
  if (!stateStr) {
    return;
  }
  PyTuple_SetItem(args.get(), 0, stateStr.release());

  auto result =
      makePyOwned(PyObject_CallObject(updateContextFunc_, args.get()));

  if (!result) {
    PyErr_Clear(); // Clear error - context update is not critical
  }

  // Clear cache to force fresh analysis
  clearCache();
#endif
}

std::string
ContextBridge::getContextualSuggestions(const std::string &stateJson) {
  if (!isAvailable() || !getSuggestionsFunc_) {
    return R"({"suggestions": []})";
  }

#ifdef PYTHON_AVAILABLE
  bridge::PyGILGuard gil; // must hold GIL for all Py* calls
  auto args = makePyOwned(PyTuple_New(1));
  if (!args) {
    return R"({"suggestions": []})";
  }
  auto stateStr = makePyOwned(PyUnicode_FromString(stateJson.c_str()));
  if (!stateStr) {
    return R"({"suggestions": []})";
  }
  PyTuple_SetItem(args.get(), 0, stateStr.release());

  auto result =
      makePyOwned(PyObject_CallObject(getSuggestionsFunc_, args.get()));

  if (!result) {
    PyErr_Clear();
    return R"({"suggestions": []})";
  }

  std::string resultStr;
  if (PyUnicode_Check(result.get())) {
    const char *cstr = PyUnicode_AsUTF8(result.get());
    resultStr = cstr ? cstr : R"({"suggestions": []})";
  } else {
    resultStr = R"({"suggestions": []})";
  }

  return resultStr;
#else
  return R"({"suggestions": []})";
#endif
}

void ContextBridge::clearCache() { cache_.clear(); }

std::string ContextBridge::hashState(const std::string &stateJson) {
  // Simple hash function for cache key
  std::hash<std::string> hasher;
  return std::to_string(hasher(stateJson));
}

} // namespace kelly
