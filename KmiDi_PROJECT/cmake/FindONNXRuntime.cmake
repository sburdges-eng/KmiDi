# FindONNXRuntime.cmake
# ---------------------
# Find ONNX Runtime library for neural network inference.
#
# This module sets the following variables:
#   ONNXRuntime_FOUND        - True if ONNX Runtime was found
#   ONNXRuntime_INCLUDE_DIRS - Include directories
#   ONNXRuntime_LIBRARIES    - Libraries to link against
#
# Usage:
#   find_package(ONNXRuntime)
#   if(ONNXRuntime_FOUND)
#       target_include_directories(MyTarget PRIVATE ${ONNXRuntime_INCLUDE_DIRS})
#       target_link_libraries(MyTarget PRIVATE ${ONNXRuntime_LIBRARIES})
#       target_compile_definitions(MyTarget PRIVATE ENABLE_ONNX_RUNTIME)
#   endif()

# Check common installation paths
find_path(ONNXRuntime_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS
        ${ONNXRuntime_ROOT}
        ${CMAKE_SOURCE_DIR}/external/onnxruntime
        /usr/local/include
        /usr/include
        /opt/homebrew/include
        $ENV{HOME}/.local/include
    PATH_SUFFIXES
        include
        include/onnxruntime
        onnxruntime
)

# Find library
find_library(ONNXRuntime_LIBRARY
    NAMES onnxruntime
    PATHS
        ${ONNXRuntime_ROOT}
        ${CMAKE_SOURCE_DIR}/external/onnxruntime
        /usr/local/lib
        /usr/lib
        /opt/homebrew/lib
        $ENV{HOME}/.local/lib
    PATH_SUFFIXES
        lib
)

# Check if we found both
if(ONNXRuntime_INCLUDE_DIR AND ONNXRuntime_LIBRARY)
    set(ONNXRuntime_FOUND TRUE)
    set(ONNXRuntime_INCLUDE_DIRS ${ONNXRuntime_INCLUDE_DIR})
    set(ONNXRuntime_LIBRARIES ${ONNXRuntime_LIBRARY})
    
    if(NOT ONNXRuntime_FIND_QUIETLY)
        message(STATUS "Found ONNX Runtime: ${ONNXRuntime_LIBRARY}")
    endif()
else()
    set(ONNXRuntime_FOUND FALSE)
    
    if(ONNXRuntime_FIND_REQUIRED)
        message(FATAL_ERROR "ONNX Runtime not found. Set ONNXRuntime_ROOT or install ONNX Runtime.")
    elseif(NOT ONNXRuntime_FIND_QUIETLY)
        message(STATUS "ONNX Runtime not found. ONNX models will not be available.")
    endif()
endif()

# Mark as advanced
mark_as_advanced(ONNXRuntime_INCLUDE_DIR ONNXRuntime_LIBRARY)

