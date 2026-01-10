#include "bridge/ContextBridge.h"

#ifdef PYTHON_AVAILABLE
#include <Python.h>
#endif

#include <sstream>
#include <functional>

namespace kelly {

ContextBridge::ContextBridge()
    : bridge::PythonBridgeBase("ContextBridge")
    , cache_(CACHE_TTL_MS, 50)
{
}

ContextBridge::~ContextBridge() {
    shutdown();
}

bool ContextBridge::initialize() {
    if (!PythonBridgeBase::initializePython()) {
        return false;
    }

    module_ = importModule("music_brain.intelligence.context_bridge");
    if (!module_) {
        return false;
    }

    analyzeContextFunc_ = getFunction(module_, "analyze_context");
    getContextualParametersFunc_ = getFunction(module_, "get_contextual_parameters");
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

std::string ContextBridge::analyzeContext(const std::string& stateJson) {
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

    PyObject* args = PyTuple_New(1);
    PyObject* stateStr = PyUnicode_FromString(stateJson.c_str());
    PyTuple_SetItem(args, 0, stateStr);

    PyObject* result = PyObject_CallObject(analyzeContextFunc_, args);
    Py_DECREF(args);

    if (!result) {
        PyErr_Print();
        return R"({"emotion_category": "unknown"})";
    }

    std::string resultStr;
    if (PyUnicode_Check(result)) {
        const char* cstr = PyUnicode_AsUTF8(result);
        resultStr = cstr ? cstr : R"({"emotion_category": "unknown"})";
    } else {
        resultStr = R"({"emotion_category": "unknown"})";
    }

    // Cache the result
    cache_.put(cacheKey, resultStr);

    Py_DECREF(result);
    return resultStr;
#else
    return R"({"emotion_category": "unknown"})";
#endif
}

std::string ContextBridge::getContextualParameters(const std::string& stateJson) {
    if (!isAvailable() || !getContextualParametersFunc_) {
        return "{}";
    }

#ifdef PYTHON_AVAILABLE
    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, PyUnicode_FromString(stateJson.c_str()));

    PyObject* result = PyObject_CallObject(getContextualParametersFunc_, args);
    Py_DECREF(args);

    if (!result) {
        PyErr_Clear();
        return "{}";
    }

    std::string resultStr;
    if (PyUnicode_Check(result)) {
        const char* cstr = PyUnicode_AsUTF8(result);
        resultStr = cstr ? cstr : "{}";
    } else {
        resultStr = "{}";
    }

    Py_DECREF(result);
    return resultStr;
#else
    return "{}";
#endif
}

void ContextBridge::updateContext(const std::string& stateJson) {
    if (!isAvailable() || !updateContextFunc_) {
        return;
    }

#ifdef PYTHON_AVAILABLE
    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, PyUnicode_FromString(stateJson.c_str()));

    PyObject* result = PyObject_CallObject(updateContextFunc_, args);
    Py_DECREF(args);

    if (result) {
        Py_DECREF(result);
    } else {
        PyErr_Clear();  // Clear error - context update is not critical
    }

    // Clear cache to force fresh analysis
    clearCache();
#endif
}

std::string ContextBridge::getContextualSuggestions(const std::string& stateJson) {
    if (!isAvailable() || !getSuggestionsFunc_) {
        return R"({"suggestions": []})";
    }

#ifdef PYTHON_AVAILABLE
    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, PyUnicode_FromString(stateJson.c_str()));

    PyObject* result = PyObject_CallObject(getSuggestionsFunc_, args);
    Py_DECREF(args);

    if (!result) {
        PyErr_Clear();
        return R"({"suggestions": []})";
    }

    std::string resultStr;
    if (PyUnicode_Check(result)) {
        const char* cstr = PyUnicode_AsUTF8(result);
        resultStr = cstr ? cstr : R"({"suggestions": []})";
    } else {
        resultStr = R"({"suggestions": []})";
    }

    Py_DECREF(result);
    return resultStr;
#else
    return R"({"suggestions": []})";
#endif
}

void ContextBridge::clearCache() {
    cache_.clear();
}

std::string ContextBridge::hashState(const std::string& stateJson) {
    // Simple hash function for cache key
    std::hash<std::string> hasher;
    return std::to_string(hasher(stateJson));
}

} // namespace kelly
