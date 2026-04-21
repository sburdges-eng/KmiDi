#include "bridge/PythonBridgeBase.h"

// Python C API - only include if Python is available
#ifdef PYTHON_AVAILABLE
#include <Python.h>
#include "bridge/PyGILGuard.h"
#endif

#include <iostream>
#include <mutex>
#include <sstream>

namespace kelly {
namespace bridge {

// Static definition of shared main-thread Python state.
// nullptr until the first successful Py_Initialize(); after that it holds
// the PyThreadState* from PyEval_SaveThread() (cast to void*).
void* PythonBridgeBase::mainThreadState_ = nullptr;

// Guards first-time PyEval_SaveThread so only one bridge calls it.
static std::once_flag s_gilReleaseOnce;

PythonBridgeBase::PythonBridgeBase(const std::string& bridgeName)
    : BridgeBase(bridgeName)
{
}

PythonBridgeBase::~PythonBridgeBase() {
    shutdownPython();
}

bool PythonBridgeBase::isPythonAvailable() {
#ifdef PYTHON_AVAILABLE
    return true;
#else
    return false;
#endif
}

bool PythonBridgeBase::isPythonInitialized() {
#ifdef PYTHON_AVAILABLE
    return Py_IsInitialized() != 0;
#else
    return false;
#endif
}

bool PythonBridgeBase::initializePython() {
#ifdef PYTHON_AVAILABLE
    // Check if Python is already initialized
    if (!Py_IsInitialized()) {
        Py_Initialize();
        if (!Py_IsInitialized()) {
            logError("Failed to initialize Python interpreter");
            return false;
        }
        pythonInitializedByThis_ = true;
    }

    // After Py_Initialize the calling thread holds the GIL.  Release it via
    // PyEval_SaveThread exactly once across all bridges so that subsequent
    // threads can acquire it with PyGILState_Ensure.  This must only happen
    // once globally; further callers already find Python initialized and must
    // instead use PyGILGuard before touching any Py* API.
    std::call_once(s_gilReleaseOnce, []() {
        // PyEval_InitThreads is a no-op in CPython >=3.9 (always called by
        // Py_Initialize) but harmless to call on older versions.
#if PY_VERSION_HEX < 0x03090000
        PyEval_InitThreads();
#endif
        mainThreadState_ = static_cast<void*>(PyEval_SaveThread());
    });

    return true;
#else
    logError("Python not available (compiled without PYTHON_AVAILABLE)");
    return false;
#endif
}

void PythonBridgeBase::shutdownPython() {
#ifdef PYTHON_AVAILABLE
    {
        // Must hold GIL before Py_DECREF on any managed object.
        PyGILGuard gil;
        for (PyObject* obj : managedObjects_) {
            if (obj) {
                Py_DECREF(obj);
            }
        }
    }
    managedObjects_.clear();

    // Only finalize if we initialized it.
    // Note: In practice, we usually don't finalize Python as other bridges
    // may still be using it. This is handled at application shutdown.
    // if (pythonInitializedByThis_ && Py_IsInitialized()) {
    //     // Restore main thread state before finalizing.
    //     if (mainThreadState_) {
    //         PyEval_RestoreThread(static_cast<PyThreadState*>(mainThreadState_));
    //         mainThreadState_ = nullptr;
    //     }
    //     Py_Finalize();
    //     pythonInitializedByThis_ = false;
    // }
#endif
}

PyObject* PythonBridgeBase::importModule(const std::string& moduleName) {
#ifdef PYTHON_AVAILABLE
    if (!isPythonInitialized()) {
        logError("Python not initialized, cannot import module: " + moduleName);
        return nullptr;
    }

    PyGILGuard gil;
    PyObject* module = PyImport_ImportModule(moduleName.c_str());
    if (!module) {
        logError("Failed to import Python module: " + moduleName);
        PyErr_Print();
        return nullptr;
    }

    registerManagedObject(module);
    return module;
#else
    logError("Python not available, cannot import module: " + moduleName);
    return nullptr;
#endif
}

PyObject* PythonBridgeBase::getFunction(PyObject* module, const std::string& funcName) {
#ifdef PYTHON_AVAILABLE
    if (!module) {
        logError("Cannot get function from null module: " + funcName);
        return nullptr;
    }

    PyGILGuard gil;
    PyObject* func = PyObject_GetAttrString(module, funcName.c_str());
    if (!func || !PyCallable_Check(func)) {
        logError("Failed to get callable function: " + funcName);
        if (func) {
            Py_DECREF(func);
        }
        return nullptr;
    }

    registerManagedObject(func);
    return func;
#else
    (void)module;
    (void)funcName;
    return nullptr;
#endif
}

std::string PythonBridgeBase::callPythonFunction(PyObject* func) {
    return callPythonFunction(func, {});
}

std::string PythonBridgeBase::callPythonFunction(PyObject* func, const std::vector<std::string>& args) {
#ifdef PYTHON_AVAILABLE
    if (!func) {
        logError("Cannot call null Python function");
        return "";
    }

    PyGILGuard gil;  // acquire GIL for all Python work in this function

    if (!PyCallable_Check(func)) {
        logError("Object is not callable");
        return "";
    }

    // Create argument tuple
    PyObject* pyArgs = PyTuple_New(static_cast<Py_ssize_t>(args.size()));
    if (!pyArgs) {
        logError("Failed to create argument tuple");
        return "";
    }

    // Add string arguments
    for (size_t i = 0; i < args.size(); ++i) {
        PyObject* argStr = PyUnicode_FromString(args[i].c_str());
        if (!argStr) {
            Py_DECREF(pyArgs);
            logError("Failed to create string argument " + std::to_string(i));
            return "";
        }
        PyTuple_SetItem(pyArgs, static_cast<Py_ssize_t>(i), argStr);
    }

    // Call the function
    PyObject* result = PyObject_CallObject(func, pyArgs);
    Py_DECREF(pyArgs);

    if (!result) {
        logError("Python function call failed");
        PyErr_Print();
        return "";
    }

    // Convert result to string
    std::string resultStr;
    if (PyUnicode_Check(result)) {
        const char* cstr = PyUnicode_AsUTF8(result);
        resultStr = cstr ? cstr : "";
    } else if (PyBytes_Check(result)) {
        const char* cstr = PyBytes_AsString(result);
        resultStr = cstr ? cstr : "";
    } else {
        // Try to convert to string representation
        PyObject* strObj = PyObject_Str(result);
        if (strObj) {
            const char* cstr = PyUnicode_AsUTF8(strObj);
            resultStr = cstr ? cstr : "";
            Py_DECREF(strObj);
        }
    }

    Py_DECREF(result);
    return resultStr;
#else
    (void)func;
    (void)args;
    return "";
#endif
}

void PythonBridgeBase::safeDecref(PyObject* obj) {
#ifdef PYTHON_AVAILABLE
    if (obj) {
        PyGILGuard gil;
        Py_DECREF(obj);
    }
#else
    (void)obj;
#endif
}

void PythonBridgeBase::registerManagedObject(PyObject* obj) {
    if (obj) {
        managedObjects_.push_back(obj);
    }
}

} // namespace bridge
} // namespace kelly
