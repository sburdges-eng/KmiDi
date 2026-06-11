# Python bindings using pybind11

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

pybind11_add_module(penta_core_native
    bindings.cpp
    harmony_bindings.cpp
    groove_bindings.cpp
    diagnostics_bindings.cpp
    osc_bindings.cpp
    ml_bindings.cpp
    prrot_bindings.cpp
)

target_link_libraries(penta_core_native PRIVATE
    penta_core
    # prrot_bindings.cpp binds prrot::*; the subsystem lives in its own
    # archive and KellyCore is optional here (BUILD_KELLY_CORE=OFF lanes).
    prrot_core
    # JUCE is linked PRIVATE by penta_core — do NOT re-link here.
    # Direct juce:: linkage was removed to prevent duplicate JUCE / ODR
    # violation / allocator mismatch in the Python process.
    # See AGENTS.md § "Duplicate JUCE / ODR / allocator mismatch".
    $<$<TARGET_EXISTS:KellyCore>:KellyCore>
    # KellyCore (static) contains PythonBridgeBase.cpp / StateBridge.cpp which
    # reference the Python C API (_Py_Initialize, _PyUnicode_FromString,
    # __Py_Dealloc, etc.). pybind11_add_module only links Python::Module
    # (headers + dynamic-lookup); the full interpreter symbols must be
    # explicitly linked here so the .so loads cleanly when Python imports it.
    Python3::Python
)

# PRROT bindings need access to src/ headers (PRROTEngine, VoiceProfile, etc.)
target_include_directories(penta_core_native PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

# Set output directory for Python module
set_target_properties(penta_core_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
)

# Install Python module
install(TARGETS penta_core_native
    LIBRARY DESTINATION python
)
