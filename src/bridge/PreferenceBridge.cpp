#include "PreferenceBridge.h"
#include "bridge/PythonBridgeBase.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <juce_core/juce_core.h>
#include <sstream>

// Python C API - only include if Python is available
#ifdef PYTHON_AVAILABLE
#include "bridge/PyGILGuard.h"
#include <Python.h>
#endif

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

// JSON string escape helper for safe serialization
static juce::String escapeJsonString(const juce::String &s) {
  return s.replace("\\", "\\\\")
      .replace("\"", "\\\"")
      .replace("\n", "\\n")
      .replace("\r", "\\r")
      .replace("\t", "\\t");
}

// Worker thread for processing Python calls
class PreferenceBridge::WorkerThread : public juce::Thread {
public:
  explicit WorkerThread(PreferenceBridge &bridge)
      : juce::Thread("PreferenceBridgeWorker"), bridge_(bridge) {}

  void run() override {
    while (!threadShouldExit()) {
      bridge_.processPendingOperations();
      bridge_.updateCachedStatistics();
      wait(500); // Check every 500ms
    }
  }

private:
  PreferenceBridge &bridge_;
};

PreferenceBridge::PreferenceBridge() {
  workerThread_ = std::make_unique<WorkerThread>(*this);
}

PreferenceBridge::~PreferenceBridge() { shutdown(); }

bool PreferenceBridge::initialize() {
  if (initialized_.load()) {
    return true;
  }

  // Initialize Python if available
  bool pythonOk = initializePython();

  if (pythonOk) {
    initialized_.store(true);
    workerThread_->startThread();
    return true;
  } else {
    // Fallback: use file-based queue (works without Python embedding)
    // This allows the system to work even if Python isn't embedded
    initialized_.store(true);
    workerThread_->startThread();
    return true;
  }
}

void PreferenceBridge::shutdown() {
  if (!initialized_.load()) {
    return;
  }

  shutdownRequested_.store(true);

  if (workerThread_ && workerThread_->isThreadRunning()) {
    workerThread_->stopThread(1000);
  }

  flush(); // Flush any pending operations
  shutdownPython();
  initialized_.store(false);
}

bool PreferenceBridge::initializePython() {
#ifdef PYTHON_AVAILABLE
  // Delegate Py_Initialize + PyEval_SaveThread to the shared, thread-safe
  // helper.  This eliminates the race where two bridges constructed on
  // different threads both observe !Py_IsInitialized() and both call
  // PyEval_SaveThread (second call is UB).
  if (!bridge::PythonBridgeBase::ensurePythonStarted()) {
    return false;
  }

  bridge::PyGILGuard gil;

  auto moduleName = makePyOwned(
      PyUnicode_FromString("music_brain.learning.user_preferences"));
  if (!moduleName)
    return false;

  auto module = makePyOwned(PyImport_Import(moduleName.get()));
  if (!module) {
    PyErr_Print();
    return false;
  }

  pythonModule_ = module.get();

  auto modelClass =
      makePyOwned(PyObject_GetAttrString(module.get(), "UserPreferenceModel"));
  if (!modelClass) {
    pythonModule_ = nullptr;
    PyErr_Print();
    return false;
  }

  auto args = makePyOwned(PyTuple_New(0));
  if (!args) {
    pythonModule_ = nullptr;
    return false;
  }
  auto instance =
      makePyOwned(PyObject_CallObject(modelClass.get(), args.get()));

  if (!instance) {
    pythonModule_ = nullptr;
    PyErr_Print();
    return false;
  }

  preferenceModel_ = instance.release();
  module.release();
  return true;
#else
  // Python not available - use file-based fallback
  return false;
#endif
}

void PreferenceBridge::shutdownPython() {
#ifdef PYTHON_AVAILABLE
  if (Py_IsInitialized()) {
    bridge::PyGILGuard
        gil; // safe: interpreter is alive; must hold GIL for Py_DECREF
    if (preferenceModel_) {
      Py_DECREF(static_cast<PyObject *>(preferenceModel_));
      preferenceModel_ = nullptr;
    }
    if (pythonModule_) {
      Py_DECREF(static_cast<PyObject *>(pythonModule_));
      pythonModule_ = nullptr;
    }
  } else {
    // Interpreter already finalized — just null the pointers; Py_DECREF is UB
    // here.
    preferenceModel_ = nullptr;
    pythonModule_ = nullptr;
  }
#endif
}

void PreferenceBridge::recordParameterAdjustment(
    const juce::String &parameterName, float oldValue, float newValue,
    const std::map<std::string, std::string> &context) {
  if (!initialized_.load())
    return;

  std::lock_guard<std::mutex> lock(pendingMutex_);
  if (pendingOperations_.size() >= MAX_PENDING_OPERATIONS) {
    // Drop oldest operation
    pendingOperations_.erase(pendingOperations_.begin());
  }

  PendingOperation op;
  op.type = PendingOperation::ParameterAdjustment;
  op.data["parameter_name"] = parameterName.toStdString();
  op.data["old_value"] = juce::String(oldValue).toStdString();
  op.data["new_value"] = juce::String(newValue).toStdString();
  for (const auto &[key, value] : context) {
    op.data["context_" + key] = value;
  }

  pendingOperations_.push_back(op);
}

void PreferenceBridge::recordEmotionSelection(
    const juce::String &emotionName, float valence, float arousal,
    const std::map<std::string, std::string> &context) {
  if (!initialized_.load())
    return;

  std::lock_guard<std::mutex> lock(pendingMutex_);
  if (pendingOperations_.size() >= MAX_PENDING_OPERATIONS) {
    pendingOperations_.erase(pendingOperations_.begin());
  }

  PendingOperation op;
  op.type = PendingOperation::EmotionSelection;
  op.data["emotion_name"] = emotionName.toStdString();
  op.data["valence"] = juce::String(valence).toStdString();
  op.data["arousal"] = juce::String(arousal).toStdString();
  for (const auto &[key, value] : context) {
    op.data["context_" + key] = value;
  }

  pendingOperations_.push_back(op);
}

juce::String PreferenceBridge::recordMidiGeneration(
    const juce::String &intentText,
    const std::map<juce::String, float> &parameters,
    const juce::String &emotion, const std::vector<juce::String> &ruleBreaks) {
  if (!initialized_.load())
    return {};

  juce::String generationId = generateId();

  std::lock_guard<std::mutex> lock(pendingMutex_);
  if (pendingOperations_.size() >= MAX_PENDING_OPERATIONS) {
    pendingOperations_.erase(pendingOperations_.begin());
  }

  PendingOperation op;
  op.type = PendingOperation::MidiGeneration;
  op.data["generation_id"] = generationId.toStdString();
  op.data["intent_text"] = intentText.toStdString();
  op.data["emotion"] = emotion.toStdString();

  // Serialize parameters
  std::ostringstream paramsStream;
  bool first = true;
  for (const auto &[key, value] : parameters) {
    if (!first)
      paramsStream << ",";
    paramsStream << key.toStdString() << "=" << value;
    first = false;
  }
  op.data["parameters"] = paramsStream.str();

  // Serialize rule breaks
  std::ostringstream rbStream;
  first = true;
  for (const auto &rb : ruleBreaks) {
    if (!first)
      rbStream << ",";
    rbStream << rb.toStdString();
    first = false;
  }
  op.data["rule_breaks"] = rbStream.str();

  pendingOperations_.push_back(op);
  return generationId;
}

void PreferenceBridge::recordMidiFeedback(const juce::String &generationId,
                                          bool accepted) {
  if (!initialized_.load())
    return;

  std::lock_guard<std::mutex> lock(pendingMutex_);
  if (pendingOperations_.size() >= MAX_PENDING_OPERATIONS) {
    pendingOperations_.erase(pendingOperations_.begin());
  }

  PendingOperation op;
  op.type = PendingOperation::MidiFeedback;
  op.data["generation_id"] = generationId.toStdString();
  op.data["accepted"] = accepted ? "true" : "false";

  pendingOperations_.push_back(op);
}

void PreferenceBridge::recordMidiModification(const juce::String &generationId,
                                              const juce::String &parameterName,
                                              float oldValue, float newValue) {
  if (!initialized_.load())
    return;

  std::lock_guard<std::mutex> lock(pendingMutex_);
  if (pendingOperations_.size() >= MAX_PENDING_OPERATIONS) {
    pendingOperations_.erase(pendingOperations_.begin());
  }

  PendingOperation op;
  op.type = PendingOperation::MidiModification;
  op.data["generation_id"] = generationId.toStdString();
  op.data["parameter_name"] = parameterName.toStdString();
  op.data["old_value"] = juce::String(oldValue).toStdString();
  op.data["new_value"] = juce::String(newValue).toStdString();

  pendingOperations_.push_back(op);
}

void PreferenceBridge::recordRuleBreakModification(
    const juce::String &ruleBreak, const juce::String &action,
    const std::map<std::string, std::string> &context) {
  if (!initialized_.load())
    return;

  std::lock_guard<std::mutex> lock(pendingMutex_);
  if (pendingOperations_.size() >= MAX_PENDING_OPERATIONS) {
    pendingOperations_.erase(pendingOperations_.begin());
  }

  PendingOperation op;
  op.type = PendingOperation::RuleBreakModification;
  op.data["rule_break"] = ruleBreak.toStdString();
  op.data["action"] = action.toStdString();
  for (const auto &[key, value] : context) {
    op.data["context_" + key] = value;
  }

  pendingOperations_.push_back(op);
}

void PreferenceBridge::processPendingOperations() {
  std::vector<PendingOperation> ops;
  {
    std::lock_guard<std::mutex> lock(pendingMutex_);
    if (pendingOperations_.empty()) {
      return;
    }
    ops.swap(pendingOperations_);
  }

#ifdef PYTHON_AVAILABLE
  if (preferenceModel_) {
    // Process with Python - each call requires the GIL.
    bridge::PyGILGuard gil;
    for (const auto &op : ops) {
      switch (op.type) {
      case PendingOperation::ParameterAdjustment: {
        float oldVal = std::stof(op.data.at("old_value"));
        float newVal = std::stof(op.data.at("new_value"));
        // Call: preferenceModel.record_parameter_adjustment(...)
        // Implementation would use PyObject_CallMethod
        (void)oldVal;
        (void)newVal;
        break;
      }
      // ... other cases
      default:
        break;
      }
    }
  } else
#endif
  {
    // File-based fallback: write to queue file
    juce::File queueFile =
        juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory)
            .getChildFile("Kelly")
            .getChildFile("preference_queue.json");

    queueFile.getParentDirectory().createDirectory();

    // Read existing queue
    juce::String existingContent;
    if (queueFile.existsAsFile()) {
      existingContent = queueFile.loadFileAsString();
    }

    // Append new operations (simplified JSON format)
    std::ostringstream jsonStream;
    jsonStream << "[\n";
    bool first = true;
    for (const auto &op : ops) {
      if (!first)
        jsonStream << ",\n";
      jsonStream << "  {\n";
      jsonStream << "    \"type\": " << static_cast<int>(op.type) << ",\n";
      jsonStream << "    \"data\": {\n";
      bool firstData = true;
      for (const auto &[key, value] : op.data) {
        if (!firstData)
          jsonStream << ",\n";
        jsonStream << "      \"" << escapeJsonString(key).toStdString()
                   << "\": \"" << escapeJsonString(value).toStdString() << "\"";
        firstData = false;
      }
      jsonStream << "\n    }\n";
      jsonStream << "  }";
      first = false;
    }
    jsonStream << "\n]";

    // Write to file (append mode for simplicity)
    std::ofstream file(queueFile.getFullPathName().toStdString(),
                       std::ios::app);
    if (file.is_open()) {
      file << jsonStream.str();
      file.close();
    }
  }
}

void PreferenceBridge::updateCachedStatistics() {
  auto now = juce::Time::currentTimeMillis();
  if (now - lastCacheUpdate_ < CACHE_UPDATE_INTERVAL_MS) {
    return;
  }

  // Update cache from Python (or file-based system)
  // For now, leave empty - will be populated when Python integration is
  // complete
  lastCacheUpdate_ = now;
}

std::map<std::string, PreferenceBridge::ParameterStatistics>
PreferenceBridge::getParameterStatistics() {
  std::lock_guard<std::mutex> lock(cacheMutex_);
  return cachedStats_;
}

std::map<std::string, int> PreferenceBridge::getEmotionPreferences() {
  std::lock_guard<std::mutex> lock(cacheMutex_);
  return cachedEmotionPrefs_;
}

std::map<std::string, std::pair<float, float>>
PreferenceBridge::getPreferredParameterRanges() {
  std::lock_guard<std::mutex> lock(cacheMutex_);
  return cachedRanges_;
}

float PreferenceBridge::getAcceptanceRate() {
  std::lock_guard<std::mutex> lock(cacheMutex_);
  return cachedAcceptanceRate_;
}

void PreferenceBridge::flush() { processPendingOperations(); }

juce::String PreferenceBridge::generateId() {
  auto time = juce::Time::currentTimeMillis();
  auto random = juce::Random::getSystemRandom().nextInt64();
  return juce::String(time) + "_" + juce::String(random);
}

bool PreferenceBridge::callPythonMethod(
    const char *methodName, const std::vector<std::string> &args,
    const std::map<std::string, std::string> &kwargs) {
#ifdef PYTHON_AVAILABLE
  if (!preferenceModel_)
    return false;

  bridge::PyGILGuard gil; // must hold GIL for all Py* calls

  auto method = makePyOwned(PyObject_GetAttrString(
      static_cast<PyObject *>(preferenceModel_), methodName));
  if (!method) {
    PyErr_Print();
    return false;
  }

  auto pyArgs = makePyOwned(PyTuple_New(args.size()));
  if (!pyArgs) {
    return false;
  }
  for (size_t i = 0; i < args.size(); ++i) {
    auto arg = makePyOwned(PyUnicode_FromString(args[i].c_str()));
    if (!arg) {
      return false;
    }
    PyTuple_SetItem(pyArgs.get(), i, arg.release());
  }

  auto result = makePyOwned(PyObject_CallObject(method.get(), pyArgs.get()));

  if (!result) {
    PyErr_Print();
    return false;
  }

  return true;
#else
  return false;
#endif
}

bool PreferenceBridge::callPythonMethodWithResult(
    const char *methodName, void *resultOut,
    const std::vector<std::string> &args,
    const std::map<std::string, std::string> &kwargs) {
  // Implementation would extract result from PyObject
  // For now, return false
  return false;
}

} // namespace kelly
