cmake_minimum_required(VERSION 3.22)
project(penta_core VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Core C++ library
cmake_minimum_required(VERSION 3.20)
project(KmiDiPentaCore LANGUAGES CXX)

# macOS SDK compatibility fix - Define function-style availability macros
# These must be defined before any system headers are included
# We define them as function-style macros (with ...) that expand to nothing
if(APPLE)
    # Note: CMAKE cannot pass function-style macros via -D, so we rely on
    # JUCE's juce_StandardHeader.h which is included first via juce_core
    # The main CMakeLists.txt sets MAC_OS_X_VERSION_MIN_REQUIRED which helps
endif()

# ReaderWriterQueue: header-only dependency (offline-only, no network fetch).
if(NOT TARGET readerwriterqueue)
    set(READERWRITERQUEUE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/readerwriterqueue")
    if(NOT EXISTS "${READERWRITERQUEUE_INCLUDE_DIR}/readerwriterqueue.h"
       AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../KmiDi_FINAL/engine/include/readerwriterqueue.h")
        set(READERWRITERQUEUE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../KmiDi_FINAL/engine/include")
    endif()

    if(NOT EXISTS "${READERWRITERQUEUE_INCLUDE_DIR}/readerwriterqueue.h")
        message(FATAL_ERROR
            "readerwriterqueue dependency missing. "
            "Provide local header at third_party/readerwriterqueue/readerwriterqueue.h")
    endif()

    set(readerwriterqueue_SOURCE_DIR "${READERWRITERQUEUE_INCLUDE_DIR}")
    add_library(readerwriterqueue INTERFACE)
    target_include_directories(readerwriterqueue INTERFACE
        ${READERWRITERQUEUE_INCLUDE_DIR}
    )
endif()

if(NOT DEFINED READERWRITERQUEUE_INCLUDE_DIR)
    get_target_property(READERWRITERQUEUE_INCLUDE_DIR readerwriterqueue INTERFACE_INCLUDE_DIRECTORIES)
endif()

# ============================================================================
# Penta-Core Library
# ============================================================================

# Collect all source files
file(GLOB_RECURSE PENTA_CORE_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/common/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/diagnostics/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/groove/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/harmony/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mixer/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ml/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/osc/*.cpp
)

# ODR consolidation (T1 audit-rt-ffi-safety-2026q2, 2026-04-21):
# Resolved 7 of 14 duplicate pairs. src_penta-core/ is canonical for all
# penta:: namespace symbols. Remaining files excluded to prevent double-definition:
#
#   VoiceLeading.cpp: src/ canonical (4 methods present only in src/; confirmed
#     by docs/CODEAUDIT_ODR_PAIR_DIFF_2026Q2.md — src_penta-core/ copy removed
#     in audit/crit-fixes-2026-04-13). src/ version compiled into KellyCore.
#   OSCClient/Hub/Server: src_penta-core/ copies removed in audit/crit-fixes-2026-04-13;
#     src/ versions with RT-safe callback model (loopback check, RealtimeCallback)
#     compile into KellyCore.
#   ScaleDetector: moved to src_penta-core/ (byte-identical; penta_core/ is canonical
#     per T1 consolidation). KellyCore exclusion added in root CMakeLists.txt.
list(FILTER PENTA_CORE_SOURCES EXCLUDE REGEX "/harmony/VoiceLeading\\.cpp$")
list(FILTER PENTA_CORE_SOURCES EXCLUDE REGEX "/osc/OSCClient\\.cpp$")
list(FILTER PENTA_CORE_SOURCES EXCLUDE REGEX "/osc/OSCHub\\.cpp$")
list(FILTER PENTA_CORE_SOURCES EXCLUDE REGEX "/osc/OSCServer\\.cpp$")

# Create the library
add_library(penta_core STATIC ${PENTA_CORE_SOURCES})

# Include directories
target_include_directories(penta_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${READERWRITERQUEUE_INCLUDE_DIR}
)

# Link libraries
# Note: juce::juce_dsp is optional - we'll make it conditional
find_package(PkgConfig QUIET)
if(TARGET juce::juce_dsp)
    target_link_libraries(penta_core PUBLIC
        readerwriterqueue
        juce::juce_dsp
    )
else()
    # Build without JUCE if not available
    message(WARNING "JUCE not available, building penta_core without juce_dsp")
    target_link_libraries(penta_core PUBLIC
        readerwriterqueue
    )
endif()

# Platform-specific settings
if(APPLE)
    target_compile_definitions(penta_core PUBLIC
        MAC_OS_X_VERSION_MIN_REQUIRED=101200
        MAC_OS_VERSION_15_0=150000
    )
endif()

# Compiler warnings
target_compile_options(penta_core PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)

# Check for AVX2 support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-mavx2 -mfma" COMPILER_SUPPORTS_AVX2)

if(COMPILER_SUPPORTS_AVX2)
    message(STATUS "AVX2 available, enabling SIMD optimizations")
    target_compile_definitions(penta_core PUBLIC ENABLE_AVX2)
    target_compile_options(penta_core PRIVATE -mavx2 -mfma)
else()
    message(STATUS "AVX2 not available, using scalar fallback")
endif()

# ============================================================================
# Tests (Optional)
# ============================================================================
option(BUILD_PENTA_TESTS "Build penta-core tests" ON)

if(BUILD_PENTA_TESTS)
    enable_testing()

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../external/googletest/CMakeLists.txt")
        add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../external/googletest"
                         "${CMAKE_CURRENT_BINARY_DIR}/external-googletest"
                         EXCLUDE_FROM_ALL)
    endif()

    if(TARGET GTest::gtest AND TARGET GTest::gtest_main)
        set(_GTEST_LIB GTest::gtest)
        set(_GTEST_MAIN_LIB GTest::gtest_main)
    elseif(TARGET gtest AND TARGET gtest_main)
        set(_GTEST_LIB gtest)
        set(_GTEST_MAIN_LIB gtest_main)
    else()
        message(FATAL_ERROR
            "BUILD_PENTA_TESTS=ON requires local googletest. "
            "Provide external/googletest or pre-install GTest.")
    endif()

    # Create test executable
    file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/../tests/penta_core/*.cpp
    )
    if("${TEST_SOURCES}" STREQUAL "")
        message(FATAL_ERROR
            "BUILD_PENTA_TESTS=ON but no test sources were found under tests/penta_core.")
    endif()

    add_executable(penta_tests ${TEST_SOURCES})

    target_link_libraries(penta_tests PRIVATE
        penta_core
        ${_GTEST_LIB}
        ${_GTEST_MAIN_LIB}
    )

    include(GoogleTest)
    gtest_discover_tests(penta_tests)
endif()
