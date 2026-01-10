# CMakeLists.txt for Android native library
# Builds JNI library linking Penta Core and JUCE Android bridge

cmake_minimum_required(VERSION 3.22)
project(idaw_android LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Android-specific settings
set(CMAKE_ANDROID_ARCH_ABI ${ANDROID_ABI})
set(CMAKE_ANDROID_NDK ${ANDROID_NDK})

# Include directories
set(PENTA_CORE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../..)
set(INCLUDE_DIRS
    ${PENTA_CORE_ROOT}/include
    ${PENTA_CORE_ROOT}/external/JUCE/modules
    ${CMAKE_CURRENT_SOURCE_DIR}
)

include_directories(${INCLUDE_DIRS})

# Penta Core source files
set(PENTA_CORE_SOURCES
    ${PENTA_CORE_ROOT}/src_penta-core/common/RTLogger.cpp
    ${PENTA_CORE_ROOT}/src_penta-core/common/RTMemoryPool.cpp
    ${PENTA_CORE_ROOT}/src_penta-core/harmony/HarmonyEngine.cpp
    ${PENTA_CORE_ROOT}/src_penta-core/harmony/ChordAnalyzer.cpp
    ${PENTA_CORE_ROOT}/src_penta-core/harmony/ScaleDetector.cpp
    ${PENTA_CORE_ROOT}/src_penta-core/harmony/VoiceLeading.cpp
    ${PENTA_CORE_ROOT}/src_penta-core/groove/GrooveEngine.cpp
    ${PENTA_CORE_ROOT}/src_penta-core/groove/OnsetDetector.cpp
    ${PENTA_CORE_ROOT}/src_penta-core/groove/RhythmQuantizer.cpp
    ${PENTA_CORE_ROOT}/src_penta-core/groove/TempoEstimator.cpp
    ${PENTA_CORE_ROOT}/src_penta-core/diagnostics/DiagnosticsEngine.cpp
    ${PENTA_CORE_ROOT}/src_penta-core/diagnostics/AudioAnalyzer.cpp
    ${PENTA_CORE_ROOT}/src_penta-core/diagnostics/PerformanceMonitor.cpp
)

# Android JNI library sources
set(ANDROID_SOURCES
    juce_android_bridge.cpp
    penta_core_wrapper.cpp
)

# Create shared library (JNI library)
add_library(idaw_android SHARED
    ${ANDROID_SOURCES}
    ${PENTA_CORE_SOURCES}
)

# Compile definitions
target_compile_definitions(idaw_android PRIVATE
    PENTA_ANDROID=1
    PENTA_NO_AVX=1
    JUCE_ANDROID=1
    ANDROID=1
    __ANDROID_API__=${ANDROID_PLATFORM_LEVEL}
)

# Compile options
target_compile_options(idaw_android PRIVATE
    -Wall
    -Wextra
    -Wno-unused-parameter
    -fPIC
    -ffast-math
    -O3
)

# Link libraries
target_link_libraries(idaw_android
    log          # Android logging
    android      # Android system library
)

# Set library output name
set_target_properties(idaw_android PROPERTIES
    LIBRARY_OUTPUT_NAME "idaw_android"
    POSITION_INDEPENDENT_CODE ON
)

# Android-specific: Disable SIMD on Android (use scalar fallback)
# Note: SIMD can be enabled for specific architectures if needed
if(ANDROID_ABI STREQUAL "arm64-v8a")
    # Could enable NEON here if needed
    target_compile_definitions(idaw_android PRIVATE PENTA_NO_AVX=1)
elseif(ANDROID_ABI STREQUAL "x86_64")
    # Could enable AVX here if needed, but disabled for compatibility
    target_compile_definitions(idaw_android PRIVATE PENTA_NO_AVX=1)
endif()

