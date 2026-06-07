#include "bridge/SuggestionBridge.h"

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

SuggestionBridge::SuggestionBridge()
    : bridge::PythonBridgeBase("SuggestionBridge") {}

SuggestionBridge::~SuggestionBridge() { shutdown(); }

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

  if (!getSuggestionsFunc_ || !recordShownFunc_ || !recordAcceptedFunc_ ||
      !recordDismissedFunc_) {
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

std::string
SuggestionBridge::getSuggestions(const std::string &currentStateJson,
                                 int maxSuggestions) {
  if (!isAvailable() || !getSuggestionsFunc_) {
    return "[]";
  }

#ifdef PYTHON_AVAILABLE
  bridge::PyGILGuard gil; // must hold GIL for all Py* calls
  auto args = makePyOwned(PyTuple_New(2));
  if (!args) {
    return "[]";
  }
  auto stateStr = makePyOwned(PyUnicode_FromString(currentStateJson.c_str()));
  auto maxSuggestionsInt = makePyOwned(PyLong_FromLong(maxSuggestions));
  if (!stateStr || !maxSuggestionsInt) {
    return "[]";
  }

  PyTuple_SetItem(args.get(), 0, stateStr.release());
  PyTuple_SetItem(args.get(), 1, maxSuggestionsInt.release());

  auto result =
      makePyOwned(PyObject_CallObject(getSuggestionsFunc_, args.get()));

  if (!result) {
    PyErr_Print();
    return "[]";
  }

  std::string resultStr;
  if (PyUnicode_Check(result.get())) {
    const char *cstr = PyUnicode_AsUTF8(result.get());
    resultStr = cstr ? cstr : "[]";
  } else {
    resultStr = "[]";
  }

  return resultStr;
#else
  return "[]";
#endif
}

void SuggestionBridge::recordSuggestionShown(const std::string &suggestionId,
                                             const std::string &suggestionType,
                                             const std::string &contextJson) {
  if (!isAvailable() || !recordShownFunc_)
    return;

#ifdef PYTHON_AVAILABLE
  bridge::PyGILGuard gil; // must hold GIL for all Py* calls
  auto args = makePyOwned(PyTuple_New(3));
  if (!args)
    return;
  auto suggestionIdObj =
      makePyOwned(PyUnicode_FromString(suggestionId.c_str()));
  auto suggestionTypeObj =
      makePyOwned(PyUnicode_FromString(suggestionType.c_str()));
  auto contextJsonObj = makePyOwned(PyUnicode_FromString(contextJson.c_str()));
  if (!suggestionIdObj || !suggestionTypeObj || !contextJsonObj)
    return;
  PyTuple_SetItem(args.get(), 0, suggestionIdObj.release());
  PyTuple_SetItem(args.get(), 1, suggestionTypeObj.release());
  PyTuple_SetItem(args.get(), 2, contextJsonObj.release());

  auto result = makePyOwned(PyObject_CallObject(recordShownFunc_, args.get()));

  if (!result) {
    PyErr_Clear(); // Clear error - tracking is not critical
  }
#endif
}

void SuggestionBridge::recordSuggestionAccepted(
    const std::string &suggestionId) {
  if (!isAvailable() || !recordAcceptedFunc_)
    return;

#ifdef PYTHON_AVAILABLE
  bridge::PyGILGuard gil; // must hold GIL for all Py* calls
  auto args = makePyOwned(PyTuple_New(1));
  if (!args)
    return;
  auto suggestionIdObj =
      makePyOwned(PyUnicode_FromString(suggestionId.c_str()));
  if (!suggestionIdObj)
    return;
  PyTuple_SetItem(args.get(), 0, suggestionIdObj.release());

  auto result =
      makePyOwned(PyObject_CallObject(recordAcceptedFunc_, args.get()));

  if (!result) {
    PyErr_Clear(); // Clear error - tracking is not critical
  }
#endif
}

void SuggestionBridge::recordSuggestionDismissed(
    const std::string &suggestionId) {
  if (!isAvailable() || !recordDismissedFunc_)
    return;

#ifdef PYTHON_AVAILABLE
  bridge::PyGILGuard gil; // must hold GIL for all Py* calls
  auto args = makePyOwned(PyTuple_New(1));
  if (!args)
    return;
  auto suggestionIdObj =
      makePyOwned(PyUnicode_FromString(suggestionId.c_str()));
  if (!suggestionIdObj)
    return;
  PyTuple_SetItem(args.get(), 0, suggestionIdObj.release());

  auto result =
      makePyOwned(PyObject_CallObject(recordDismissedFunc_, args.get()));

  if (!result) {
    PyErr_Clear(); // Clear error - tracking is not critical
  }
#endif
}

} // namespace kelly
