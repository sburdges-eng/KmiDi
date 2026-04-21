#pragma once
/**
 * PyGILGuard.h - RAII helper for CPython GIL acquisition
 * =======================================================
 *
 * Any C++ thread that is NOT the main Python thread must hold the GIL before
 * touching any PyObject*, calling any Py* API function, or adjusting reference
 * counts (Py_DECREF / Py_INCREF / Py_XDECREF).
 *
 * Usage (within a #ifdef PYTHON_AVAILABLE block):
 *
 *   void MyWorkerThread::doWork() {
 *   #ifdef PYTHON_AVAILABLE
 *       bridge::PyGILGuard gil;   // acquire once at the top
 *       PyObject* result = PyObject_CallObject(func_, args_);
 *       // ... all Python work here ...
 *       Py_DECREF(result);
 *   #endif
 *   }
 *
 * The guard is NOT necessary inside Py_Initialize() itself (the main thread
 * owns the GIL implicitly before PyEval_SaveThread releases it). It IS
 * necessary in every subsequent call from any thread, including the main
 * thread after PyEval_SaveThread has been called.
 *
 * Safe to nest: PyGILState_Ensure / PyGILState_Release handle re-entrancy.
 */

#ifdef PYTHON_AVAILABLE
#include <Python.h>

namespace kelly {
namespace bridge {

// RAII acquisition of the CPython GIL.  Construct in any thread that is
// about to touch a PyObject, call any Python C API function, or decrement
// a reference count.  Safe to nest (PyGILState machinery tracks recursion).
struct PyGILGuard {
    PyGILState_STATE state;
    PyGILGuard()  : state(PyGILState_Ensure()) {}
    ~PyGILGuard() { PyGILState_Release(state); }
    PyGILGuard(const PyGILGuard&)            = delete;
    PyGILGuard& operator=(const PyGILGuard&) = delete;
};

} // namespace bridge
} // namespace kelly

#endif // PYTHON_AVAILABLE
