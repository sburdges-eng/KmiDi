# FindRTNeural.cmake
# -------------------
# Find RTNeural library for real-time neural network inference.
#
# This module sets the following variables:
#   RTNeural_FOUND        - True if RTNeural was found
#   RTNeural_INCLUDE_DIRS - Include directories for RTNeural
#   RTNeural_LIBRARIES    - Libraries to link against (header-only, so empty)
#
# Usage:
#   find_package(RTNeural)
#   if(RTNeural_FOUND)
#       target_include_directories(MyTarget PRIVATE ${RTNeural_INCLUDE_DIRS})
#       target_compile_definitions(MyTarget PRIVATE ENABLE_RTNEURAL)
#   endif()

# RTNeural is header-only, so we just need to find the include directory

# Check common installation paths
find_path(RTNeural_INCLUDE_DIR
    NAMES RTNeural/RTNeural.h
    PATHS
        ${RTNeural_ROOT}
        ${CMAKE_SOURCE_DIR}/external/RTNeural
        /usr/local/include
        /usr/include
        /opt/homebrew/include
        $ENV{HOME}/.local/include
    PATH_SUFFIXES
        include
)

# Check if we found it
if(RTNeural_INCLUDE_DIR)
    set(RTNeural_FOUND TRUE)
    set(RTNeural_INCLUDE_DIRS ${RTNeural_INCLUDE_DIR})
    set(RTNeural_LIBRARIES "")  # Header-only
    
    if(NOT RTNeural_FIND_QUIETLY)
        message(STATUS "Found RTNeural: ${RTNeural_INCLUDE_DIR}")
    endif()
else()
    set(RTNeural_FOUND FALSE)
    
    if(RTNeural_FIND_REQUIRED)
        message(FATAL_ERROR "RTNeural not found. Set RTNeural_ROOT or install RTNeural.")
    elseif(NOT RTNeural_FIND_QUIETLY)
        message(STATUS "RTNeural not found. ML will use fallback heuristics.")
    endif()
endif()

# Mark as advanced
mark_as_advanced(RTNeural_INCLUDE_DIR)

