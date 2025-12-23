#include "bridge/SuggestionBridge.h"

#ifdef PYTHON_AVAILABLE
#include <Python.h>
#endif

#include <sstream>

namespace kelly {

SuggestionBridge::SuggestionBridge()
    : bridge::PythonBridgeBase("SuggestionBridge")
{
}

SuggestionBridge::~SuggestionBridge() {
    shutdown();
}

bool SuggestionBridge::initialize() {
    if (!PythonBridgeBase::initializePython()) {
        return false;
    }

    module_ = importModule("music_brain.intelligence.suggestion_bridge");
    if (!module_) {
        return false;
    }

    getSuggestionsFunc_ = getFunction(module_, "get_suggestions");
    recordShownFunc_ = getFunction(module_, "record_suggestion_shown");
    recordAcceptedFunc_ = getFunction(module_, "record_suggestion_accepted");
    recordDismissedFunc_ = getFunction(module_, "record_suggestion_dismissed");

    if (!getSuggestionsFunc_ || !recordShownFunc_ || !recordAcceptedFunc_ || !recordDismissedFunc_) {
        logError("Failed to get function pointers");
        return false;
    }

    setAvailable(true);
    return true;
}

void SuggestionBridge::shutdown() {
    getSuggestionsFunc_ = nullptr;
    recordShownFunc_ = nullptr;
    recordAcceptedFunc_ = nullptr;
    recordDismissedFunc_ = nullptr;
    module_ = nullptr;
    PythonBridgeBase::shutdownPython();
    setAvailable(false);
}

std::string SuggestionBridge::getSuggestions(
    const std::string& currentStateJson,
    int maxSuggestions
) {
    if (!isAvailable() || !getSuggestionsFunc_) {
        return "[]";
    }

#ifdef PYTHON_AVAILABLE
    PyObject* args = PyTuple_New(2);
    PyObject* stateStr = PyUnicode_FromString(currentStateJson.c_str());
    PyObject* maxSuggestionsInt = PyLong_FromLong(maxSuggestions);

    PyTuple_SetItem(args, 0, stateStr);
    PyTuple_SetItem(args, 1, maxSuggestionsInt);

    PyObject* result = PyObject_CallObject(getSuggestionsFunc_, args);
    Py_DECREF(args);

    if (!result) {
        PyErr_Print();
        return "[]";
    }

    std::string resultStr;
    if (PyUnicode_Check(result)) {
        const char* cstr = PyUnicode_AsUTF8(result);
        resultStr = cstr ? cstr : "[]";
    } else {
        resultStr = "[]";
    }

    Py_DECREF(result);
    return resultStr;
#else
    return "[]";
#endif
}

void SuggestionBridge::recordSuggestionShown(
    const std::string& suggestionId,
    const std::string& suggestionType,
    const std::string& contextJson
) {
    if (!isAvailable() || !recordShownFunc_) return;

#ifdef PYTHON_AVAILABLE
    PyObject* args = PyTuple_New(3);
    PyTuple_SetItem(args, 0, PyUnicode_FromString(suggestionId.c_str()));
    PyTuple_SetItem(args, 1, PyUnicode_FromString(suggestionType.c_str()));
    PyTuple_SetItem(args, 2, PyUnicode_FromString(contextJson.c_str()));

    PyObject* result = PyObject_CallObject(recordShownFunc_, args);
    Py_DECREF(args);

    if (result) {
        Py_DECREF(result);
    } else {
        PyErr_Clear();  // Clear error - tracking is not critical
    }
#endif
}

void SuggestionBridge::recordSuggestionAccepted(const std::string& suggestionId) {
    if (!isAvailable() || !recordAcceptedFunc_) return;

#ifdef PYTHON_AVAILABLE
    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, PyUnicode_FromString(suggestionId.c_str()));

    PyObject* result = PyObject_CallObject(recordAcceptedFunc_, args);
    Py_DECREF(args);

    if (result) {
        Py_DECREF(result);
    } else {
        PyErr_Clear();  // Clear error - tracking is not critical
    }
#endif
}

void SuggestionBridge::recordSuggestionDismissed(const std::string& suggestionId) {
    if (!isAvailable() || !recordDismissedFunc_) return;

#ifdef PYTHON_AVAILABLE
    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, PyUnicode_FromString(suggestionId.c_str()));

    PyObject* result = PyObject_CallObject(recordDismissedFunc_, args);
    Py_DECREF(args);

    if (result) {
        Py_DECREF(result);
    } else {
        PyErr_Clear();  // Clear error - tracking is not critical
    }
#endif
}

} // namespace kelly
