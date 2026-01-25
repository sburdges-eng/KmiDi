cmake_minimum_required(VERSION 3.27)
project(PentaCore VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set source directories
set(PENTA_CORE_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../KmiDi_PROJECT/source/cpp/src_penta-core")
set(PENTA_CORE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../KmiDi_PROJECT/include")

# Fetch dependencies
include(FetchContent)

# Readerwriterqueue (lock-free queue)
FetchContent_Declare(
    readerwriterqueue
    GIT_REPOSITORY https://github.com/cameron314/readerwriterqueue.git
    GIT_TAG v1.0.6
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(readerwriterqueue)

# nlohmann/json
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

# Define sources
set(PENTA_CORE_SOURCES
    ${PENTA_CORE_SRC_DIR}/common/RTMemoryPool.cpp
    ${PENTA_CORE_SRC_DIR}/common/RTLogger.cpp
    ${PENTA_CORE_SRC_DIR}/harmony/ChordAnalyzer.cpp
    ${PENTA_CORE_SRC_DIR}/harmony/ChordAnalyzerSIMD.cpp
    ${PENTA_CORE_SRC_DIR}/harmony/ScaleDetector.cpp
    ${PENTA_CORE_SRC_DIR}/harmony/VoiceLeading.cpp
    ${PENTA_CORE_SRC_DIR}/harmony/HarmonyEngine.cpp
    ${PENTA_CORE_SRC_DIR}/groove/OnsetDetector.cpp
    ${PENTA_CORE_SRC_DIR}/groove/TempoEstimator.cpp
    ${PENTA_CORE_SRC_DIR}/groove/RhythmQuantizer.cpp
    ${PENTA_CORE_SRC_DIR}/groove/GrooveEngine.cpp
    ${PENTA_CORE_SRC_DIR}/diagnostics/PerformanceMonitor.cpp
    ${PENTA_CORE_SRC_DIR}/diagnostics/AudioAnalyzer.cpp
    ${PENTA_CORE_SRC_DIR}/diagnostics/DiagnosticsEngine.cpp
    # OSC files excluded due to JUCE dependency
    # ${PENTA_CORE_SRC_DIR}/osc/OSCServer.cpp
    # ${PENTA_CORE_SRC_DIR}/osc/OSCClient.cpp
    # ${PENTA_CORE_SRC_DIR}/osc/OSCMessage.cpp
    # ${PENTA_CORE_SRC_DIR}/osc/RTMessageQueue.cpp
    # ${PENTA_CORE_SRC_DIR}/osc/OSCHub.cpp
    ${PENTA_CORE_SRC_DIR}/ml/MLInterface.cpp
    ${PENTA_CORE_SRC_DIR}/mixer/MixerEngine.cpp
)

# Create library
add_library(penta_core STATIC ${PENTA_CORE_SOURCES})

set_target_properties(penta_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(penta_core PUBLIC
    ${PENTA_CORE_INCLUDE_DIR}
    ${readerwriterqueue_SOURCE_DIR}
)

target_link_libraries(penta_core PUBLIC
    readerwriterqueue
    nlohmann_json::nlohmann_json
)

# Platform-specific threading
if(UNIX)
    target_link_libraries(penta_core PUBLIC pthread)
endif()

# Enable AVX2 SIMD optimizations if available
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
if(COMPILER_SUPPORTS_AVX2)
    target_compile_options(penta_core PRIVATE -mavx2 -mpopcnt)
    message(STATUS "AVX2 SIMD optimizations enabled")
else()
    message(STATUS "AVX2 not available, using scalar fallback")
endif()

# Add tests
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()

    # Fetch Catch2
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.4.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(Catch2)

    add_executable(penta_core_tests
        ${PENTA_CORE_SRC_DIR}/../../../tests/penta_core/harmony_test.cpp
        ${PENTA_CORE_SRC_DIR}/../../../tests/penta_core/groove_test.cpp
        ${PENTA_CORE_SRC_DIR}/../../../tests/penta_core/performance_test.cpp
    )

    target_link_libraries(penta_core_tests PRIVATE
        penta_core
        Catch2::Catch2WithMain
    )

    include(${catch2_SOURCE_DIR}/extras/Catch.cmake)
    catch_discover_tests(penta_core_tests)
endif()
