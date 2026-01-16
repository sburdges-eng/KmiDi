cmake_minimum_required(VERSION 3.27)
project(KmiDi VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CPP_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/source/cpp)
    set(CPP_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/source/cpp)
endif()

set(PY_PENTA_CORE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/python/penta_core)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/source/python/penta_core)
    set(PY_PENTA_CORE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/source/python/penta_core)
endif()

# Options
option(BUILD_DESKTOP "Build desktop GUI application" ON)
option(BUILD_PLUGINS "Build VST3 and CLAP plugins" ON)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_KMIDI_CORE "Build KmiDi core library/app/plugins" ON)
option(BUILD_PENTA_TESTS "Build penta_core test suite" OFF)
option(ENABLE_TRACY "Enable Tracy profiling" OFF)
option(ENABLE_RTNEURAL "Enable RTNeural for ML inference" OFF)
option(ENABLE_ONNX_RUNTIME "Enable ONNX Runtime for ML inference" OFF)

# CMake modules path
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Prefer system audio codecs; disable JUCE's bundled Ogg/Vorbis to avoid missing sources
set(JUCE_USE_OGGVORBIS FALSE CACHE BOOL "Disable JUCE Ogg/Vorbis codec" FORCE)
add_compile_definitions(JUCE_USE_OGGVORBIS=0)
# Disable VST2 replacement to avoid requiring VST2 SDK headers
add_compile_definitions(JUCE_VST3_CAN_REPLACE_VST2=0)

# Homebrew Ogg/Vorbis paths for JUCE audio_formats (macOS)
if(APPLE AND EXISTS "/opt/homebrew/include" AND EXISTS "/opt/homebrew/lib")
    include_directories("/opt/homebrew/include")
    link_directories("/opt/homebrew/lib")
endif()

# macOS SDK 26.2+ compatibility fix for __CLOCK_AVAILABILITY
# The SDK header _time.h uses __CLOCK_AVAILABILITY in enum definitions, which requires
# __OSX_AVAILABLE to be defined. We ensure deployment target is set so Availability.h works.
# Fix is implemented in external/JUCE/modules/juce_core/system/juce_StandardHeader.h
# These settings MUST be configured before any targets are created to apply effectively.
if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.12" CACHE STRING "Minimum macOS version")
    # Set minimum required version for availability macros
    # The SDK's Availability.h handles all availability attributes correctly
    add_compile_definitions(
        MAC_OS_X_VERSION_MIN_REQUIRED=101200
        MAC_OS_VERSION_15_0=150000
    )
endif()
find_package(Qt6 COMPONENTS Core Widgets REQUIRED)

# JUCE setup
add_subdirectory(external/JUCE EXCLUDE_FROM_ALL)

# Ensure juceaide helper is available (needed for plugin targets)
if(NOT TARGET juce::juceaide)
    set(JUCE_JUCEAIDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/external/JUCE/extras/Build/juceaide")
    if(EXISTS "${JUCE_JUCEAIDE_PATH}")
        add_executable(juce::juceaide IMPORTED GLOBAL)
        set_target_properties(juce::juceaide PROPERTIES IMPORTED_LOCATION "${JUCE_JUCEAIDE_PATH}")
    endif()
endif()

# Readerwriterqueue (lock-free queue) for RT-safe OSC
# Note: This is a header-only library with no CMakeLists.txt, so we must
# explicitly create an INTERFACE target and ensure SOURCE_DIR is populated.
include(FetchContent)
FetchContent_Declare(
    readerwriterqueue
    GIT_REPOSITORY https://github.com/cameron314/readerwriterqueue.git
    GIT_TAG v1.0.6
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(readerwriterqueue)

# Explicitly populate SOURCE_DIR for header-only library without CMakeLists.txt
FetchContent_GetProperties(readerwriterqueue)

if(NOT TARGET readerwriterqueue)
    add_library(readerwriterqueue INTERFACE)
    target_include_directories(readerwriterqueue INTERFACE
        ${readerwriterqueue_SOURCE_DIR}
    )
endif()

# ============================================================================
# Penta-Core Library (C++ RT engines)
# ============================================================================
add_subdirectory(${CPP_ROOT}/src_penta-core)

# ============================================================================
# Python Bindings (Optional)
# ============================================================================
option(BUILD_PYTHON_BINDINGS "Build Python bindings for penta-core" ON)

if(BUILD_PYTHON_BINDINGS)
    # Find pybind11
    find_package(pybind11 QUIET)
    if(pybind11_FOUND)
        message(STATUS "pybind11 found, building Python bindings")
        add_subdirectory(${CPP_ROOT}/bindings)
    else()
        message(STATUS "pybind11 not found. Python bindings will not be built.")
        message(STATUS "  Install via: pip install pybind11")
    endif()
endif()

# ============================================================================
# ML Dependencies (Optional)
# ============================================================================

# RTNeural - Header-only neural network library for real-time inference
if(ENABLE_RTNEURAL)
    find_package(RTNeural QUIET)
    if(RTNeural_FOUND)
        message(STATUS "RTNeural enabled: ${RTNeural_INCLUDE_DIRS}")
    else()
        # Try to fetch RTNeural if not found
        message(STATUS "RTNeural not found locally, fetching from GitHub...")
        FetchContent_Declare(
            rtneural
            GIT_REPOSITORY https://github.com/jatinchowdhury18/RTNeural.git
            GIT_TAG main
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(rtneural)
        set(RTNeural_FOUND TRUE)
        set(RTNeural_INCLUDE_DIRS ${rtneural_SOURCE_DIR})
    endif()
endif()

# ONNX Runtime - Cross-platform inference engine
if(ENABLE_ONNX_RUNTIME)
    find_package(ONNXRuntime QUIET)
    if(ONNXRuntime_FOUND)
        message(STATUS "ONNX Runtime enabled: ${ONNXRuntime_LIBRARIES}")
    else()
        message(STATUS "ONNX Runtime not found. ONNX models will not be available.")
        message(STATUS "  Install ONNX Runtime or set ONNXRuntime_ROOT")
    endif()
endif()

# ============================================================================
# Main Kelly library (shared core)
# ============================================================================
file(GLOB_RECURSE KELLY_CORE_SOURCES CONFIGURE_DEPENDS
    ${CPP_ROOT}/src/*.cpp
    ${CPP_ROOT}/src/*.mm
    ${CPP_ROOT}/src/midi/MidiIO.cpp
    ${CPP_ROOT}/src/ui/ScoreEntryPanel.cpp
    ${CPP_ROOT}/src/ui/MixerConsolePanel.cpp
)

# Exclude app entry point and plugin sources from the core lib
list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/gui/main.cpp$")
list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/plugin/")
list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/engine/test_kelly\\.cpp$")
list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/python/")  # Python bindings built separately
# Exclude RTNeural-dependent files when RTNeural is not enabled
# These files require RTNeural headers to be available
if(NOT ENABLE_RTNEURAL)
    list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/KellyML/ExampleUsage\\.cpp$")
    list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/KellyML/KellyMLModel\\.cpp$")
    list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/KellyML/KellyMLPipeline\\.cpp$")
    list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/KellyML/SnapshotWriter\\.cpp$")
endif()
# Exclude Python bridge and binding sources from the core lib when bindings are built separately
if(BUILD_PYTHON_BINDINGS)
    list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/bridge/kelly_bridge\\.cpp$")
    list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/python/bindings\\.cpp$")
endif()
# NOTE: Previous line 31 had "list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/")"
# which would exclude ALL source files (since all paths contain "/src/"), making
# KELLY_CORE_SOURCES empty. This has been removed. If macro conflicts exist,
# they should be fixed by excluding specific problematic files/directories rather
# than excluding everything.

if(BUILD_KMIDI_CORE)
add_library(KellyCore STATIC ${KELLY_CORE_SOURCES})

# Ensure ObjC++ file (for biometrics) is treated correctly if present
set_source_files_properties(
    ${CPP_ROOT}/src/biometric/BiometricInput.mm
    PROPERTIES COMPILE_LANGUAGES OBJCXX
)

target_include_directories(KellyCore PUBLIC
    ${CPP_ROOT}/src
    ${CPP_ROOT}/include
    ${readerwriterqueue_SOURCE_DIR}
)

target_link_libraries(KellyCore PUBLIC
    Qt6::Core
    Qt6::Widgets
    juce::juce_audio_basics
    juce::juce_audio_devices
    juce::juce_audio_formats
    juce::juce_audio_processors
    juce::juce_core
    juce::juce_data_structures
    juce::juce_events
    juce::juce_graphics
    juce::juce_gui_basics
    juce::juce_dsp
    juce::juce_osc
    readerwriterqueue
)

# ML Dependencies
if(ENABLE_RTNEURAL AND RTNeural_FOUND)
    target_include_directories(KellyCore PUBLIC ${RTNeural_INCLUDE_DIRS})
    target_compile_definitions(KellyCore PUBLIC ENABLE_RTNEURAL)
    message(STATUS "KellyCore: RTNeural inference enabled")
endif()

if(ENABLE_ONNX_RUNTIME AND ONNXRuntime_FOUND)
    target_include_directories(KellyCore PUBLIC ${ONNXRuntime_INCLUDE_DIRS})
    target_link_libraries(KellyCore PUBLIC ${ONNXRuntime_LIBRARIES})
    target_compile_definitions(KellyCore PUBLIC ENABLE_ONNX_RUNTIME)
    message(STATUS "KellyCore: ONNX Runtime inference enabled")
endif()

target_compile_options(KellyCore PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)

# Kelly GUI Application
if(BUILD_DESKTOP)
    if(NOT BUILD_KMIDI_CORE)
        message(FATAL_ERROR "BUILD_DESKTOP requires BUILD_KMIDI_CORE=ON")
    endif()
    add_executable(KellyApp
        ${CPP_ROOT}/src/gui/main.cpp
        ${CPP_ROOT}/src/gui/main_window.cpp
    )

    set_target_properties(KellyApp PROPERTIES
        AUTOMOC ON
    )

    target_link_libraries(KellyApp PRIVATE
        KellyCore
        Qt6::Core
        Qt6::Widgets
    )
endif()

# Plugin builds
if(BUILD_PLUGINS)
    if(NOT BUILD_KMIDI_CORE)
        message(FATAL_ERROR "BUILD_PLUGINS requires BUILD_KMIDI_CORE=ON")
    endif()
    # VST3/CLAP plugin
    juce_add_plugin(KellyPlugin
        COMPANY_NAME "KmiDi"
        PLUGIN_MANUFACTURER_CODE Klly
        PLUGIN_CODE Klp1
        FORMATS VST3 CLAP
        PRODUCT_NAME "KmiDi Emotion Processor"
        VST3_CATEGORIES Fx Synth
        CLAP_ID com.kelly.emotion-processor
    )

    target_sources(KellyPlugin PRIVATE
        ${CPP_ROOT}/src/plugin/PluginProcessor.cpp
        ${CPP_ROOT}/src/plugin/PluginEditor.cpp
        ${CPP_ROOT}/src/plugin/PluginState.cpp
        ${CPP_ROOT}/src/plugin/MasterEQProcessor.cpp
    )

    target_link_libraries(KellyPlugin PRIVATE
        KellyCore
        juce::juce_audio_plugin_client
    )
endif()

endif() # BUILD_KMIDI_CORE

# Tests
if(BUILD_TESTS AND BUILD_KMIDI_CORE)
    enable_testing()
    # Fetch Catch2 if it doesn't exist locally
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.4.0 # Use a specific tag for stability
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(Catch2)
    include(${catch2_SOURCE_DIR}/extras/Catch.cmake)

    add_executable(KellyTests
        tests/cpp/test_emotion_engine.cpp
        tests/cpp/test_midi_pipeline.cpp
        tests/cpp/test_chord_diagnostics.cpp
        tests/cpp/test_ml_pipeline.cpp
        benchmarks/harmony_latency.cpp
        benchmarks/groove_latency.cpp
    )
    target_link_libraries(KellyTests PRIVATE
        KellyCore
        Catch2::Catch2WithMain
    )
    include(CTest)
    catch_discover_tests(KellyTests)
endif()

if(BUILD_PENTA_TESTS)
    enable_testing()
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(googletest)
    add_subdirectory(tests/penta_core)
endif()

# Tracy profiling
if(ENABLE_TRACY)
    add_subdirectory(external/tracy EXCLUDE_FROM_ALL)
    target_link_libraries(KellyCore PUBLIC TracyClient)
    target_compile_definitions(KellyCore PUBLIC TRACY_ENABLE)
endif()

# Installation
if(BUILD_KMIDI_CORE)
    set(KELLY_INSTALL_TARGETS KellyCore)
    if(BUILD_DESKTOP)
        list(APPEND KELLY_INSTALL_TARGETS KellyApp)
    endif()

    install(TARGETS ${KELLY_INSTALL_TARGETS}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )

    install(DIRECTORY ${CPP_ROOT}/src/
        DESTINATION include/kelly
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
    )

    # Install ML models
    install(DIRECTORY models/
        DESTINATION share/kelly/models
        FILES_MATCHING
        PATTERN "*.json"
        PATTERN "*.onnx"
    )

    # Install Python ML module (optional)
    if(EXISTS ${PY_PENTA_CORE_DIR})
        install(DIRECTORY ${PY_PENTA_CORE_DIR}
            DESTINATION share/kelly/python
            FILES_MATCHING PATTERN "*.py"
        )
    endif()
endif()
