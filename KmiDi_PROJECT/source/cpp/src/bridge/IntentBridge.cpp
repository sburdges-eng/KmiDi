#include "bridge/IntentBridge.h"

#ifdef PYTHON_AVAILABLE
#include <Python.h>
#endif

#include <sstream>
#include <regex>

namespace kelly {

IntentBridge::IntentBridge()
    : bridge::PythonBridgeBase("IntentBridge")
    , cache_(CACHE_TTL_MS, 50)
{
}

IntentBridge::~IntentBridge() {
    shutdown();
}

bool IntentBridge::initialize() {
    if (!PythonBridgeBase::initializePython()) {
        return false;
    }

    module_ = importModule("music_brain.session.intent_bridge");
    if (!module_) {
        return false;
    }

    processIntentFunc_ = getFunction(module_, "process_intent");
    convertToCppFunc_ = getFunction(module_, "convert_to_cpp_intent");
    convertToPythonFunc_ = getFunction(module_, "convert_to_python_intent");
    validateResultFunc_ = getFunction(module_, "validate_result");
    getRuleBreaksFunc_ = getFunction(module_, "get_suggested_rule_breaks");

    // Not all functions are required - processIntent is the main one
    if (!processIntentFunc_) {
        logError("Failed to get process_intent function");
        return false;
    }

    setAvailable(true);
    return true;
}

void IntentBridge::shutdown() {
    processIntentFunc_ = nullptr;
    convertToCppFunc_ = nullptr;
    convertToPythonFunc_ = nullptr;
    validateResultFunc_ = nullptr;
    getRuleBreaksFunc_ = nullptr;
    module_ = nullptr;
    cache_.clear();
    PythonBridgeBase::shutdownPython();
    setAvailable(false);
}

std::string IntentBridge::processIntent(const std::string& intentJson) {
    if (!isAvailable()) {
        return "{}";
    }

    // Check cache first
    std::string cacheKey = hashIntent(intentJson);
    std::string cached = cache_.get(cacheKey);
    if (!cached.empty()) {
        return cached;
    }

#ifdef PYTHON_AVAILABLE
    if (!processIntentFunc_) {
        return "{}";
    }

    PyObject* args = PyTuple_New(1);
    PyObject* intentStr = PyUnicode_FromString(intentJson.c_str());
    PyTuple_SetItem(args, 0, intentStr);

    PyObject* result = PyObject_CallObject(processIntentFunc_, args);
    Py_DECREF(args);

    if (!result) {
        PyErr_Print();
        return "{}";
    }

    std::string resultStr;
    if (PyUnicode_Check(result)) {
        const char* cstr = PyUnicode_AsUTF8(result);
        resultStr = cstr ? cstr : "{}";
    } else {
        resultStr = "{}";
    }

    // Cache the result
    cache_.put(cacheKey, resultStr);

    Py_DECREF(result);
    return resultStr;
#else
    return "{}";
#endif
}

IntentResult IntentBridge::convertToCppIntent(const std::string& intentJson) {
    std::string resultJson = processIntent(intentJson);
    return parseIntentResult(resultJson);
}

std::string IntentBridge::convertToPythonIntent(const IntentResult& intent) {
    if (!isAvailable() || !convertToPythonFunc_) {
        // Fallback: create basic Python intent format
        std::ostringstream oss;
        oss << R"({"phase_1": {"mood_primary": ")" << intent.sourceWound.primaryEmotion.name
            << R"(", "vulnerability_scale": 0.5}, "phase_2": {"technical_key": ")"
            << intent.key << R"(", "technical_mode": ")" << intent.mode << R"("}})";
        return oss.str();
    }

#ifdef PYTHON_AVAILABLE
    // Convert IntentResult to JSON first (simplified)
    std::ostringstream intentJson;
    intentJson << R"({"key": ")" << intent.key
               << R"(", "mode": ")" << intent.mode
               << R"(", "tempoBpm": )" << intent.tempoBpm
               << R"(, "emotion": ")" << intent.sourceWound.primaryEmotion.name << R"("})";

    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, PyUnicode_FromString(intentJson.str().c_str()));

    PyObject* result = PyObject_CallObject(convertToPythonFunc_, args);
    Py_DECREF(args);

    if (!result) {
        PyErr_Clear();
        return intentJson.str();
    }

    std::string resultStr;
    if (PyUnicode_Check(result)) {
        const char* cstr = PyUnicode_AsUTF8(result);
        resultStr = cstr ? cstr : intentJson.str();
    } else {
        resultStr = intentJson.str();
    }

    Py_DECREF(result);
    return resultStr;
#else
    return "{}";
#endif
}

bool IntentBridge::validateResult(const std::string& resultJson) {
    if (!isAvailable() || !validateResultFunc_) {
        // Basic validation: check if it's valid JSON structure
        return !resultJson.empty() && resultJson != "{}";
    }

#ifdef PYTHON_AVAILABLE
    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, PyUnicode_FromString(resultJson.c_str()));

    PyObject* result = PyObject_CallObject(validateResultFunc_, args);
    Py_DECREF(args);

    if (!result) {
        PyErr_Clear();
        return false;
    }

    bool valid = PyObject_IsTrue(result);
    Py_DECREF(result);
    return valid;
#else
    return false;
#endif
}

std::string IntentBridge::getSuggestedRuleBreaks(const std::string& emotion) {
    if (!isAvailable() || !getRuleBreaksFunc_) {
        return R"({"rule_breaks": []})";
    }

#ifdef PYTHON_AVAILABLE
    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, PyUnicode_FromString(emotion.c_str()));

    PyObject* result = PyObject_CallObject(getRuleBreaksFunc_, args);
    Py_DECREF(args);

    if (!result) {
        PyErr_Clear();
        return R"({"rule_breaks": []})";
    }

    std::string resultStr;
    if (PyUnicode_Check(result)) {
        const char* cstr = PyUnicode_AsUTF8(result);
        resultStr = cstr ? cstr : R"({"rule_breaks": []})";
    } else {
        resultStr = R"({"rule_breaks": []})";
    }

    Py_DECREF(result);
    return resultStr;
#else
    return R"({"rule_breaks": []})";
#endif
}


std::string IntentBridge::hashIntent(const std::string& intentJson) {
    // Simple hash function for cache key
    std::hash<std::string> hasher;
    return std::to_string(hasher(intentJson));
}

IntentResult IntentBridge::parseIntentResult(const std::string& resultJson) {
    IntentResult result;

    // Simple JSON parsing (in production, use a proper JSON library)
    // For now, extract key fields using regex
    // Using delimiter to handle quotes in raw string literals
    std::regex keyRegex(R"delim("key"\s*:\s*"([^"]+)")delim");
    std::regex modeRegex(R"delim("mode"\s*:\s*"([^"]+)")delim");
    std::regex tempoRegex(R"delim("tempoBpm"\s*:\s*(\d+))delim");
    std::regex emotionRegex(R"delim("emotion"\s*:\s*"([^"]+)")delim");

    std::smatch match;
    if (std::regex_search(resultJson, match, keyRegex)) {
        result.key = match[1].str();
    }
    if (std::regex_search(resultJson, match, modeRegex)) {
        result.mode = match[1].str();
    }
    if (std::regex_search(resultJson, match, tempoRegex)) {
        result.tempoBpm = std::stoi(match[1].str());
    }

    // Parse chord progression
    std::regex chordRegex(R"delim("chordProgression"\s*:\s*\[(.*?)\])delim");
    if (std::regex_search(resultJson, match, chordRegex)) {
        std::string chordsStr = match[1].str();
        std::regex chordNameRegex(R"delim("([^"]+)")delim");
        std::sregex_iterator iter(chordsStr.begin(), chordsStr.end(), chordNameRegex);
        std::sregex_iterator end;
        for (; iter != end; ++iter) {
            result.chordProgression.push_back((*iter)[1].str());
        }
    }

    // Default values if not found
    if (result.key.empty()) result.key = "C";
    if (result.mode.empty()) result.mode = "major";
    if (result.tempoBpm == 0) result.tempoBpm = 120;

    return result;
}

} // namespace kelly
