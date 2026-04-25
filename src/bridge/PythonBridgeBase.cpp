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

// Single once_flag that wraps Py_Initialize + PyEval_SaveThread together so
// that the same thread that acquires the GIL via Py_Initialize is also the
// one that releases it via PyEval_SaveThread. Splitting these across two
// synchronization points lets thread A initialize (acquiring the GIL) while
// thread B then races into PyEval_SaveThread without holding it — UB.
static std::once_flag s_pyStartupOnce;

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

// static
bool PythonBridgeBase::ensurePythonStarted() {
#ifdef PYTHON_AVAILABLE
    // Both Py_Initialize and PyEval_SaveThread must run on the same thread
    // and run together — the first half acquires the GIL, the second half
    // releases it. Wrapping both inside one call_once means exactly one
    // thread races to do the whole startup; every other thread blocks until
    // it completes, then sees mainThreadState_ already populated.
    static std::atomic<bool> startupOk{false};
    std::call_once(s_pyStartupOnce, []() {
        if (!Py_IsInitialized()) {
            Py_Initialize();
        }
        if (!Py_IsInitialized()) {
            std::cerr << "PythonBridgeBase: Failed to initialize Python interpreter\n";
            return;
        }
        // PyEval_InitThreads is a no-op in CPython >=3.9 (Py_Initialize calls
        // it implicitly) but harmless on older versions.
#if PY_VERSION_HEX < 0x03090000
        PyEval_InitThreads();
#endif
        // Releasing the GIL here is paired with the Py_Initialize above on
        // the same thread — required by CPython.
        mainThreadState_ = static_cast<void*>(PyEval_SaveThread());
        startupOk.store(true, std::memory_order_release);
    });
    return startupOk.load(std::memory_order_acquire);
#else
    std::cerr << "PythonBridgeBase: Python not available (compiled without PYTHON_AVAILABLE)\n";
    return false;
#endif
}

bool PythonBridgeBase::initializePython() {
#ifdef PYTHON_AVAILABLE
    if (!ensurePythonStarted()) {
        logError("Failed to initialize Python interpreter");
        return false;
    }
    if (!Py_IsInitialized()) {
        // ensurePythonStarted returned true; this branch is unreachable but
        // kept for defensive correctness.
        return false;
    }
    // Track whether this instance was the one that first initialized Python
    // (used by shutdownPython to decide whether to call Py_Finalize).
    // The once_flag ensures only one bridge ever calls Py_Initialize, so
    // pythonInitializedByThis_ is only meaningful for ownership tracking.
    pythonInitializedByThis_ = true;
    // If Python was already initialized before ensurePythonStarted(), the
    // GIL has already been released.  Do NOT touch thread-state here.
    return true;
#else
    logError("Python not available (compiled without PYTHON_AVAILABLE)");
    return false;
#endif
}

void PythonBridgeBase::shutdownPython() {
#ifdef PYTHON_AVAILABLE
    if (!managedObjects_.empty() && Py_IsInitialized()) {
        PyGILGuard gil;  // safe: interpreter is alive
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
    if (obj && Py_IsInitialized()) {
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
