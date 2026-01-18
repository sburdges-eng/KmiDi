cmake_minimum_required(VERSION 3.27)
project(KmiDi VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# KmiDi_FINAL consolidated structure paths
set(ENGINE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/engine)
set(PY_PENTA_CORE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/python/penta_core)
set(PLUGINS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/plugins)
set(SHARED_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/shared)
set(ML_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/ml)
set(BUILD_CONFIG_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/build)

# Options
option(BUILD_DESKTOP "Build desktop GUI application" ON)
option(BUILD_PLUGINS "Build VST3 and CLAP plugins" ON)
option(BUILD_TESTS "Build tests" OFF)
option(BUILD_KMIDI_CORE "Build KmiDi core library/app/plugins" ON)
option(BUILD_PENTA_TESTS "Build penta_core test suite" OFF)
option(ENABLE_TRACY "Enable Tracy profiling" OFF)
option(ENABLE_RTNEURAL "Enable RTNeural for ML inference" OFF)
option(ENABLE_ONNX_RUNTIME "Enable ONNX Runtime for ML inference" OFF)

# CMake modules path
list(APPEND CMAKE_MODULE_PATH ${BUILD_CONFIG_ROOT}/cmake)

# Prefer system audio codecs; disable JUCE's bundled Ogg/Vorbis
set(JUCE_USE_OGGVORBIS FALSE CACHE BOOL "Disable JUCE Ogg/Vorbis codec" FORCE)
add_compile_definitions(JUCE_USE_OGGVORBIS=0)
add_compile_definitions(JUCE_VST3_CAN_REPLACE_VST2=0)

# Homebrew paths (macOS)
if(APPLE AND EXISTS "/opt/homebrew/include" AND EXISTS "/opt/homebrew/lib")
    include_directories("/opt/homebrew/include")
    link_directories("/opt/homebrew/lib")
endif()

# macOS SDK compatibility
if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.12" CACHE STRING "Minimum macOS version")
    add_compile_definitions(
        MAC_OS_X_VERSION_MIN_REQUIRED=101200
        MAC_OS_VERSION_15_0=150000
    )
endif()

find_package(Qt6 COMPONENTS Core Widgets REQUIRED)

# JUCE setup - use consolidated build/external/JUCE
add_subdirectory(build/external/JUCE EXCLUDE_FROM_ALL)

# Ensure juceaide helper is available
if(NOT TARGET juce::juceaide)
    set(JUCE_JUCEAIDE_PATH "${BUILD_CONFIG_ROOT}/external/JUCE/extras/Build/juceaide")
    if(EXISTS "${JUCE_JUCEAIDE_PATH}")
        add_executable(juce::juceaide IMPORTED GLOBAL)
        set_target_properties(juce::juceaide PROPERTIES IMPORTED_LOCATION "${JUCE_JUCEAIDE_PATH}")
    endif()
endif()

# Readerwriterqueue (lock-free queue)
include(FetchContent)
FetchContent_Declare(
    readerwriterqueue
    GIT_REPOSITORY https://github.com/cameron314/readerwriterqueue.git
    GIT_TAG v1.0.6
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(readerwriterqueue)
FetchContent_GetProperties(readerwriterqueue)

if(NOT TARGET readerwriterqueue)
    add_library(readerwriterqueue INTERFACE)
    target_include_directories(readerwriterqueue INTERFACE ${readerwriterqueue_SOURCE_DIR})
endif()

# ============================================================================
# Penta-Core Library (C++ RT engines)
# ============================================================================
add_subdirectory(engine/src_penta-core)

# ============================================================================
# Python Bindings (Optional)
# ============================================================================
option(BUILD_PYTHON_BINDINGS "Build Python bindings for penta-core" ON)

if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 QUIET)
    if(pybind11_FOUND)
        message(STATUS "pybind11 found, building Python bindings")
        add_subdirectory(engine/bindings)
    else()
        message(STATUS "pybind11 not found. Python bindings will not be built.")
    endif()
endif()

# ============================================================================
# ML Dependencies (Optional)
# ============================================================================
if(ENABLE_RTNEURAL)
    find_package(RTNeural QUIET)
    if(NOT RTNeural_FOUND)
        message(STATUS "RTNeural not found locally, fetching from GitHub...")
        FetchContent_Declare(rtneural
            GIT_REPOSITORY https://github.com/jatinchowdhury18/RTNeural.git
            GIT_TAG main
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(rtneural)
        set(RTNeural_FOUND TRUE)
        set(RTNeural_INCLUDE_DIRS ${rtneural_SOURCE_DIR})
    endif()
endif()

if(ENABLE_ONNX_RUNTIME)
    find_package(ONNXRuntime QUIET)
    if(ONNXRuntime_FOUND)
        message(STATUS "ONNX Runtime enabled: ${ONNXRuntime_LIBRARIES}")
    else()
        message(STATUS "ONNX Runtime not found.")
    endif()
endif()

# ============================================================================
# Main Kelly library (shared core)
# ============================================================================
file(GLOB_RECURSE KELLY_CORE_SOURCES CONFIGURE_DEPENDS
    ${ENGINE_ROOT}/src/*.cpp
    ${ENGINE_ROOT}/src/*.mm
)

# Exclude app entry point and plugin sources from the core lib
list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/gui/main.cpp$")
list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/plugin/")
list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/engine/test_kelly\\.cpp$")
list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/python/")

if(NOT ENABLE_RTNEURAL)
    list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/KellyML/ExampleUsage\\.cpp$")
    list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/KellyML/KellyMLModel\\.cpp$")
    list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/KellyML/KellyMLPipeline\\.cpp$")
    list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/KellyML/SnapshotWriter\\.cpp$")
endif()

if(BUILD_PYTHON_BINDINGS)
    list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/bridge/kelly_bridge\\.cpp$")
    list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/python/bindings\\.cpp$")
endif()

if(BUILD_KMIDI_CORE)

add_library(KellyCore STATIC ${KELLY_CORE_SOURCES})

set_source_files_properties(
    ${ENGINE_ROOT}/src/biometric/BiometricInput.mm
    PROPERTIES COMPILE_LANGUAGES OBJCXX
)

target_include_directories(KellyCore PUBLIC
    ${ENGINE_ROOT}/src
    ${ENGINE_ROOT}/include
    ${SHARED_ROOT}/include
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

if(ENABLE_RTNEURAL AND RTNeural_FOUND)
    target_include_directories(KellyCore PUBLIC ${RTNeural_INCLUDE_DIRS})
    target_compile_definitions(KellyCore PUBLIC ENABLE_RTNEURAL)
endif()

if(ENABLE_ONNX_RUNTIME AND ONNXRuntime_FOUND)
    target_include_directories(KellyCore PUBLIC ${ONNXRuntime_INCLUDE_DIRS})
    target_link_libraries(KellyCore PUBLIC ${ONNXRuntime_LIBRARIES})
    target_compile_definitions(KellyCore PUBLIC ENABLE_ONNX_RUNTIME)
endif()

target_compile_options(KellyCore PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)

# Kelly GUI Application
if(BUILD_DESKTOP)
    add_executable(KellyApp
        ${ENGINE_ROOT}/src/gui/main.cpp
        ${ENGINE_ROOT}/src/gui/main_window.cpp
    )
    set_target_properties(KellyApp PROPERTIES AUTOMOC ON)
    target_link_libraries(KellyApp PRIVATE KellyCore Qt6::Core Qt6::Widgets)
    if(APPLE)
        set(KMIDI_APP_ICON "${CMAKE_CURRENT_SOURCE_DIR}/assets/icons/app/KmiDi.icns")
        set_source_files_properties(${KMIDI_APP_ICON} PROPERTIES MACOSX_PACKAGE_LOCATION "Resources")
        target_sources(KellyApp PRIVATE ${KMIDI_APP_ICON})
        set_target_properties(KellyApp PROPERTIES
            MACOSX_BUNDLE TRUE
            MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/apps/macOS/Info.plist"
            MACOSX_BUNDLE_ICON_FILE "KmiDi.icns"
        )
    endif()
endif()

# Plugin builds
if(BUILD_PLUGINS)
    juce_add_plugin(KellyPlugin
        COMPANY_NAME "KmiDi"
        PLUGIN_MANUFACTURER_CODE Klly
        PLUGIN_CODE Klp1
        FORMATS VST3 AU CLAP Standalone
        PRODUCT_NAME "KmiDi Emotion Processor"
        VST3_CATEGORIES Fx Synth
        NEEDS_MIDI_INPUT TRUE
        NEEDS_MIDI_OUTPUT TRUE
        IS_MIDI_EFFECT TRUE
        VST_NUM_MIDI_INS 1
        VST_NUM_MIDI_OUTS 1
        AU_MAIN_TYPE kAudioUnitType_MIDIProcessor
        CLAP_ID com.kelly.emotion-processor
        ICON_BIG "${CMAKE_CURRENT_SOURCE_DIR}/assets/icons/plugin/KmiDi_plugin_256.png"
        ICON_SMALL "${CMAKE_CURRENT_SOURCE_DIR}/assets/icons/plugin/KmiDi_plugin_32.png"
    )

    target_sources(KellyPlugin PRIVATE
        ${PLUGINS_ROOT}/PluginProcessor.cpp
        ${PLUGINS_ROOT}/PluginEditor.cpp
        ${PLUGINS_ROOT}/PluginState.cpp
        ${PLUGINS_ROOT}/MasterEQProcessor.cpp
    )

    target_link_libraries(KellyPlugin PRIVATE
        KellyCore
        juce::juce_audio_plugin_client
        juce::juce_audio_utils
        juce::juce_audio_devices
    )
endif()

endif() # BUILD_KMIDI_CORE

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
        BUNDLE DESTINATION bin
    )

    install(DIRECTORY ${ENGINE_ROOT}/src/
        DESTINATION include/kelly
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
    )

    install(DIRECTORY ${ML_ROOT}/models/
        DESTINATION share/kelly/models
        FILES_MATCHING
        PATTERN "*.json"
        PATTERN "*.onnx"
        PATTERN "*.mlpackage"
    )

    if(EXISTS ${PY_PENTA_CORE_DIR})
        install(DIRECTORY ${PY_PENTA_CORE_DIR}
            DESTINATION share/kelly/python
            FILES_MATCHING PATTERN "*.py"
        )
    endif()
endif()
