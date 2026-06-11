#include "bridge/EngineIntelligenceBridge.h"

#ifdef PYTHON_AVAILABLE
#include "bridge/PyGILGuard.h"
#include <Python.h>
#endif

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

EngineIntelligenceBridge::EngineIntelligenceBridge()
    : bridge::PythonBridgeBase("EngineIntelligenceBridge"),
      cache_(CACHE_TTL_MS, 100) {}

EngineIntelligenceBridge::~EngineIntelligenceBridge() { shutdown(); }

bool EngineIntelligenceBridge::initialize() {
  if (!PythonBridgeBase::initializePython()) {
    return false;
  }

  module_ = importModule("music_brain.intelligence.engine_bridge");
  if (!module_) {
    return false;
  }

  getEngineSuggestionsFunc_ = getFunction(module_, "get_engine_suggestions");
  getBatchSuggestionsFunc_ =
      getFunction(module_, "get_batch_engine_suggestions");
  recordAppliedFunc_ = getFunction(module_, "record_suggestion_applied");
  reportEngineStateFunc_ = getFunction(module_, "report_engine_state");

  if (!getEngineSuggestionsFunc_) {
    logError("Failed to get function pointers");
    return false;
  }

  setAvailable(true);
  return true;
}

void EngineIntelligenceBridge::shutdown() {
  getEngineSuggestionsFunc_ = nullptr;
  getBatchSuggestionsFunc_ = nullptr;
  recordAppliedFunc_ = nullptr;
  reportEngineStateFunc_ = nullptr;
  module_ = nullptr;
  cache_.clear();
  PythonBridgeBase::shutdownPython();
  setAvailable(false);
}

std::string EngineIntelligenceBridge::getEngineSuggestions(
    const std::string &engineType, const std::string &currentStateJson) {
  if (!isAvailable()) {
    return R"({"parameter_adjustments": {}, "style_suggestions": [], "rule_breaks": [], "confidence": 0.0})";
  }

  // Check cache first
  std::string cacheKey = engineType + ":" + hashState(currentStateJson);
  std::string cached = cache_.get(cacheKey);
  if (!cached.empty()) {
    return cached;
  }

#ifdef PYTHON_AVAILABLE
  if (!getEngineSuggestionsFunc_) {
    return R"({"parameter_adjustments": {}, "style_suggestions": [], "rule_breaks": [], "confidence": 0.0})";
  }

  bridge::PyGILGuard gil; // must hold GIL for all Py* calls
  auto args = makePyOwned(PyTuple_New(2));
  if (!args) {
    return R"({"parameter_adjustments": {}, "style_suggestions": [], "rule_breaks": [], "confidence": 0.0})";
  }
  auto engineTypeStr = makePyOwned(PyUnicode_FromString(engineType.c_str()));
  auto stateStr = makePyOwned(PyUnicode_FromString(currentStateJson.c_str()));
  if (!engineTypeStr || !stateStr) {
    return R"({"parameter_adjustments": {}, "style_suggestions": [], "rule_breaks": [], "confidence": 0.0})";
  }

  PyTuple_SetItem(args.get(), 0, engineTypeStr.release());
  PyTuple_SetItem(args.get(), 1, stateStr.release());

  auto result =
      makePyOwned(PyObject_CallObject(getEngineSuggestionsFunc_, args.get()));

  if (!result) {
    PyErr_Print();
    return R"({"parameter_adjustments": {}, "style_suggestions": [], "rule_breaks": [], "confidence": 0.0})";
  }

  std::string resultStr;
  if (PyUnicode_Check(result.get())) {
    const char *cstr = PyUnicode_AsUTF8(result.get());
    resultStr =
        cstr
            ? cstr
            : R"({"parameter_adjustments": {}, "style_suggestions": [], "rule_breaks": [], "confidence": 0.0})";
  } else {
    resultStr =
        R"({"parameter_adjustments": {}, "style_suggestions": [], "rule_breaks": [], "confidence": 0.0})";
  }

  cache_.put(cacheKey, resultStr);
  return resultStr;
#else
  return R"({"parameter_adjustments": {}, "style_suggestions": [], "rule_breaks": [], "confidence": 0.0})";
#endif
}

void EngineIntelligenceBridge::reportEngineState(
    const std::string &engineType, const std::string &stateJson,
    const std::string &generatedNotesJson) {
  if (!isAvailable() || !reportEngineStateFunc_)
    return;

#ifdef PYTHON_AVAILABLE
  bridge::PyGILGuard gil; // must hold GIL for all Py* calls
  auto args = makePyOwned(PyTuple_New(3));
  if (!args)
    return;
  auto engineTypeObj = makePyOwned(PyUnicode_FromString(engineType.c_str()));
  auto stateJsonObj = makePyOwned(PyUnicode_FromString(stateJson.c_str()));
  auto generatedNotesObj =
      makePyOwned(PyUnicode_FromString(generatedNotesJson.c_str()));
  if (!engineTypeObj || !stateJsonObj || !generatedNotesObj)
    return;
  PyTuple_SetItem(args.get(), 0, engineTypeObj.release());
  PyTuple_SetItem(args.get(), 1, stateJsonObj.release());
  PyTuple_SetItem(args.get(), 2, generatedNotesObj.release());

  auto result =
      makePyOwned(PyObject_CallObject(reportEngineStateFunc_, args.get()));

  if (!result) {
    PyErr_Clear(); // Clear error - state reporting is not critical
  }
#endif
}

void EngineIntelligenceBridge::clearCache() { cache_.clear(); }

std::string EngineIntelligenceBridge::hashState(const std::string &stateJson) {
  std::hash<std::string> hasher;
  return std::to_string(hasher(stateJson));
}

} // namespace kelly
