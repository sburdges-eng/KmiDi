#include "bridge/EngineIntelligenceBridge.h"

#ifdef PYTHON_AVAILABLE
#include <Python.h>
#endif

#include <sstream>

namespace kelly {

EngineIntelligenceBridge::EngineIntelligenceBridge()
    : bridge::PythonBridgeBase("EngineIntelligenceBridge")
    , cache_(CACHE_TTL_MS, 100)
{
}

EngineIntelligenceBridge::~EngineIntelligenceBridge() {
    shutdown();
}

bool EngineIntelligenceBridge::initialize() {
    if (!PythonBridgeBase::initializePython()) {
        return false;
    }

    module_ = importModule("music_brain.intelligence.engine_bridge");
    if (!module_) {
        return false;
    }

    getEngineSuggestionsFunc_ = getFunction(module_, "get_engine_suggestions");
    getBatchSuggestionsFunc_ = getFunction(module_, "get_batch_engine_suggestions");
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
    const std::string& engineType,
    const std::string& currentStateJson
) {
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

    PyObject* args = PyTuple_New(2);
    PyObject* engineTypeStr = PyUnicode_FromString(engineType.c_str());
    PyObject* stateStr = PyUnicode_FromString(currentStateJson.c_str());

    PyTuple_SetItem(args, 0, engineTypeStr);
    PyTuple_SetItem(args, 1, stateStr);

    PyObject* result = PyObject_CallObject(getEngineSuggestionsFunc_, args);
    Py_DECREF(args);

    if (!result) {
        PyErr_Print();
        return R"({"parameter_adjustments": {}, "style_suggestions": [], "rule_breaks": [], "confidence": 0.0})";
    }

    std::string resultStr;
    if (PyUnicode_Check(result)) {
        const char* cstr = PyUnicode_AsUTF8(result);
        resultStr = cstr ? cstr : R"({"parameter_adjustments": {}, "style_suggestions": [], "rule_breaks": [], "confidence": 0.0})";
    } else {
        resultStr = R"({"parameter_adjustments": {}, "style_suggestions": [], "rule_breaks": [], "confidence": 0.0})";
    }

    // Cache the result
    cache_.put(cacheKey, resultStr);

    Py_DECREF(result);
    return resultStr;
#else
    return R"({"parameter_adjustments": {}, "style_suggestions": [], "rule_breaks": [], "confidence": 0.0})";
#endif
}

void EngineIntelligenceBridge::reportEngineState(
    const std::string& engineType,
    const std::string& stateJson,
    const std::string& generatedNotesJson
) {
    if (!isAvailable() || !reportEngineStateFunc_) return;

#ifdef PYTHON_AVAILABLE
    PyObject* args = PyTuple_New(3);
    PyTuple_SetItem(args, 0, PyUnicode_FromString(engineType.c_str()));
    PyTuple_SetItem(args, 1, PyUnicode_FromString(stateJson.c_str()));
    PyTuple_SetItem(args, 2, PyUnicode_FromString(generatedNotesJson.c_str()));

    PyObject* result = PyObject_CallObject(reportEngineStateFunc_, args);
    Py_DECREF(args);

    if (result) {
        Py_DECREF(result);
    } else {
        PyErr_Clear();  // Clear error - state reporting is not critical
    }
#endif
}

void EngineIntelligenceBridge::clearCache() {
    cache_.clear();
}

std::string EngineIntelligenceBridge::hashState(const std::string& stateJson) {
    std::hash<std::string> hasher;
    return std::to_string(hasher(stateJson));
}

} // namespace kelly
