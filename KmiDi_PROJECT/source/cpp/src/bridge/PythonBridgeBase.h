#pragma once

/**
 * PythonBridgeBase.h - Base class for Python C API bridges
 * ========================================================
 *
 * Provides common Python C API functionality:
 * - Python initialization/shutdown management
 * - Module importing with error handling
 * - Function pointer retrieval and management
 * - Safe Python object reference counting
 * - Common Python call patterns
 */

#include "bridge/BridgeBase.h"
#include <string>
#include <vector>
#include <memory>

// Forward declaration to avoid including Python.h in header
struct _object;
typedef struct _object PyObject;

namespace kelly {
namespace bridge {

/**
 * PythonBridgeBase - Base class for bridges using Python C API
 *
 * Handles all common Python C API operations to reduce code duplication
 * across Python bridges.
 */
class PythonBridgeBase : public BridgeBase {
public:
    PythonBridgeBase(const std::string& bridgeName);
    virtual ~PythonBridgeBase();

    /**
     * Initialize Python interpreter (if not already initialized)
     * @return true if Python is available and initialized
     */
    bool initializePython();

    /**
     * Shutdown Python and cleanup all managed objects
     */
    void shutdownPython();

protected:
    /**
     * Import a Python module
     * @param moduleName Full module path (e.g., "music_brain.intelligence.context_bridge")
     * @return PyObject* of the module, or nullptr on error
     */
    PyObject* importModule(const std::string& moduleName);

    /**
     * Get a function from a Python module
     * @param module Module object (from importModule)
     * @param funcName Function name
     * @return PyObject* of the function, or nullptr on error
     */
    PyObject* getFunction(PyObject* module, const std::string& funcName);

    /**
     * Call a Python function with string arguments
     * @param func Function object (from getFunction)
     * @param args Vector of string arguments
     * @return Result as string, or empty string on error
     */
    std::string callPythonFunction(PyObject* func, const std::vector<std::string>& args);

    /**
     * Call a Python function with no arguments
     * @param func Function object
     * @return Result as string, or empty string on error
     */
    std::string callPythonFunction(PyObject* func);

    /**
     * Safely decrement Python object reference count
     * @param obj Object to decref (can be nullptr)
     */
    void safeDecref(PyObject* obj);

    /**
     * Check if Python is available (compiled with Python support)
     */
    static bool isPythonAvailable();

    /**
     * Check if Python interpreter is initialized
     */
    static bool isPythonInitialized();

    /**
     * Register a Python object for automatic cleanup
     * @param obj Object to manage
     */
    void registerManagedObject(PyObject* obj);

private:
    // Managed Python objects (for cleanup)
    std::vector<PyObject*> managedObjects_;
    bool pythonInitializedByThis_ = false;
};

} // namespace bridge
} // namespace kelly
