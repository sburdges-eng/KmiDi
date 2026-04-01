# FindONNXRuntime.cmake
#
# Locates the ONNX Runtime headers and library.
#
# Inputs (optional):
#   ONNXRuntime_ROOT       — path hint (CMake variable)
#   ENV{ONNXRuntime_ROOT}  — path hint (environment variable)
#
# Outputs:
#   ONNXRuntime_FOUND        — TRUE when both header and library are found
#   ONNXRuntime_INCLUDE_DIRS — directory containing onnxruntime_cxx_api.h
#   ONNXRuntime_LIBRARIES    — full path to libonnxruntime

find_path(ONNXRuntime_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS
        "${ONNXRuntime_ROOT}"
        "$ENV{ONNXRuntime_ROOT}"
        /opt/homebrew/opt/onnxruntime
        /usr/local/opt/onnxruntime
    PATH_SUFFIXES
        include/onnxruntime
        include
)

find_library(ONNXRuntime_LIBRARY
    NAMES onnxruntime
    HINTS
        "${ONNXRuntime_ROOT}"
        "$ENV{ONNXRuntime_ROOT}"
        /opt/homebrew/opt/onnxruntime
        /usr/local/opt/onnxruntime
    PATH_SUFFIXES lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNXRuntime
    REQUIRED_VARS ONNXRuntime_LIBRARY ONNXRuntime_INCLUDE_DIR
)

if(ONNXRuntime_FOUND)
    set(ONNXRuntime_INCLUDE_DIRS "${ONNXRuntime_INCLUDE_DIR}")
    set(ONNXRuntime_LIBRARIES    "${ONNXRuntime_LIBRARY}")

    if(NOT TARGET ONNXRuntime::ONNXRuntime)
        add_library(ONNXRuntime::ONNXRuntime UNKNOWN IMPORTED)
        set_target_properties(ONNXRuntime::ONNXRuntime PROPERTIES
            IMPORTED_LOCATION             "${ONNXRuntime_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${ONNXRuntime_INCLUDE_DIR}"
        )
    endif()
endif()

mark_as_advanced(ONNXRuntime_INCLUDE_DIR ONNXRuntime_LIBRARY)
