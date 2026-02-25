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

# Readerwriterqueue: header-only library
# Must explicitly create INTERFACE target and ensure SOURCE_DIR is populated
include(FetchContent)

# Always get properties in case it was already fetched by parent CMakeLists
FetchContent_GetProperties(readerwriterqueue)

if(NOT readerwriterqueue_POPULATED)
    FetchContent_Declare(
        readerwriterqueue
        GIT_REPOSITORY https://github.com/cameron314/readerwriterqueue.git
        GIT_TAG v1.0.6
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(readerwriterqueue)
endif()

# Explicitly populate SOURCE_DIR for header-only library without CMakeLists.txt
FetchContent_GetProperties(readerwriterqueue)

if(NOT TARGET readerwriterqueue)
    add_library(readerwriterqueue INTERFACE)
    target_include_directories(readerwriterqueue INTERFACE
        ${readerwriterqueue_SOURCE_DIR}
    )
endif()

# nlohmann/json for JSON parsing
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

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

# Create the library
add_library(penta_core STATIC ${PENTA_CORE_SOURCES})

# Include directories
target_include_directories(penta_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${readerwriterqueue_SOURCE_DIR}
)

# Link libraries
target_link_libraries(penta_core PUBLIC
    readerwriterqueue
    nlohmann_json::nlohmann_json
    juce::juce_dsp
)

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

    # Fetch GoogleTest
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(googletest)

    # Create test executable
    file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/../tests/penta_core/*.cpp
    )

    add_executable(penta_tests ${TEST_SOURCES})

    target_link_libraries(penta_tests PRIVATE
        penta_core
        gtest
        gtest_main
    )

    include(GoogleTest)
    gtest_discover_tests(penta_tests)
endif()
