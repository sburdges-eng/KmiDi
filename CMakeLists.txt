cmake_minimum_required(VERSION 3.27)
project(Kelly VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(BUILD_DESKTOP "Build desktop GUI application" ON)
option(BUILD_PLUGINS "Build VST3 and CLAP plugins" ON)
option(BUILD_TESTS "Build tests" OFF)
option(BUILD_KELLY_CORE "Build Kelly core library/app/plugins" ON)
option(BUILD_PENTA_TESTS "Build penta_core test suite" OFF)
option(KMIDI_BUILD_QT_UI "Build legacy Qt6 UI surfaces" OFF)
option(KMIDI_BUILD_JUCE_UI "Build legacy JUCE UI surfaces" OFF)
option(ENABLE_TRACY "Enable Tracy profiling" OFF)
option(ENABLE_RTNEURAL "Enable RTNeural for ML inference" OFF)
option(ENABLE_ONNX_RUNTIME "Enable ONNX Runtime for ML inference" OFF)
# Freeze/CI: when ON, disallow FetchContent; vendor readerwriterqueue, RTNeural, googletest in external/
option(KMIDI_OFFLINE_BUILD "Disallow network dependency resolution at configure/build time" OFF)
# Headless RT callback harness for engine (section 3.3)
option(BUILD_RT_HARNESS "Build headless RT callback harness" ON)

# KmiDi_FINAL integration options
option(USE_KMI_DI_FINAL "Use existing KmiDi_FINAL components" OFF)
option(BUILD_NATIVE_MACOS_APP "Build native macOS app from KmiDi_FINAL" OFF)

# KmiDi_FINAL paths
# Prefer the in-repo copy if present; fall back to the external sibling path used in older layouts.
set(_kmidi_final_default "${CMAKE_CURRENT_SOURCE_DIR}/KmiDi_FINAL")
if(NOT EXISTS "${_kmidi_final_default}")
    set(_kmidi_final_default "${CMAKE_SOURCE_DIR}/../KmiDi-1/KmiDi_FINAL")
endif()
set(KMI_DI_FINAL_ROOT "${_kmidi_final_default}"
    CACHE PATH "Path to KmiDi_FINAL directory")

# CMake modules path
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# KmiDi_FINAL validation and integration
if(USE_KMI_DI_FINAL OR BUILD_NATIVE_MACOS_APP)
    if(NOT EXISTS ${KMI_DI_FINAL_ROOT})
        message(WARNING "KmiDi_FINAL not found at ${KMI_DI_FINAL_ROOT}. Integration features will be disabled.")
        set(KMI_DI_FINAL_AVAILABLE OFF)
    else()
        message(STATUS "KmiDi_FINAL found at: ${KMI_DI_FINAL_ROOT}")
        set(KMI_DI_FINAL_AVAILABLE ON)
    endif()
endif()

# Pure DSP core from KmiDi_FINAL (if available)
set(_kmidi_final_dsp_dir "${KMI_DI_FINAL_ROOT}/engine/src/dsp")
set(_kmidi_final_dsp_cmake "${_kmidi_final_dsp_dir}/CMakeLists.txt")
if(USE_KMI_DI_FINAL AND KMI_DI_FINAL_AVAILABLE AND EXISTS "${_kmidi_final_dsp_cmake}")
    message(STATUS "Including pure DSP core from KmiDi_FINAL")
    add_subdirectory("${_kmidi_final_dsp_dir}" kmidi_dsp_core)
    set(KMI_DI_HAS_PURE_DSP ON)
elseif(USE_KMI_DI_FINAL AND KMI_DI_FINAL_AVAILABLE AND EXISTS "${_kmidi_final_dsp_dir}")
    message(WARNING
        "KmiDi_FINAL DSP directory exists but has no CMakeLists.txt; skipping add_subdirectory().")
endif()

# macOS SDK 26.2+ compatibility fix for __CLOCK_AVAILABILITY
# The SDK header _time.h uses __CLOCK_AVAILABILITY in enum definitions, which requires
# __OSX_AVAILABLE to be defined. We ensure deployment target is set so Availability.h works.
# Fix is implemented in external/JUCE/modules/juce_core/system/juce_StandardHeader.h
# These settings MUST be configured before any targets are created to apply effectively.
if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.12" CACHE STRING "Minimum macOS version")
    # Ensure availability macros are available before system headers
    # Use numeric value (101200 = 10.12 * 10000) since macro constants aren't available at CMake time
    # MAX_ALLOWED defaults to SDK version if not set, so we only set MIN_REQUIRED
    # Note: CMAKE_OSX_DEPLOYMENT_TARGET also sets this automatically, but we set it explicitly
    # to ensure it's available before JUCE headers are processed
    #
    # For SDK 26.2+, we define simple (non-function) availability macros to empty.
    # Function-style macros (__API_AVAILABLE, etc.) are defined in juce_StandardHeader.h
    # because CMake can't pass function-style preprocessor definitions on the command line.
    add_compile_definitions(
        MAC_OS_X_VERSION_MIN_REQUIRED=101200
        __CLOCK_AVAILABILITY=
        __WATCHOS_PROHIBITED=
        __TVOS_PROHIBITED=
        __IOS_PROHIBITED=
    )
endif()

# v1 stabilization defaults: keep legacy UI surfaces off unless explicitly requested.
if(NOT KMIDI_BUILD_QT_UI)
    set(BUILD_DESKTOP OFF CACHE BOOL "Build desktop GUI application" FORCE)
endif()

if(NOT KMIDI_BUILD_JUCE_UI)
    set(BUILD_PLUGINS OFF CACHE BOOL "Build VST3 and CLAP plugins" FORCE)
endif()

# Qt is only required when Qt-backed targets are being built.
set(KMIDI_REQUIRE_QT OFF)
if(BUILD_KELLY_CORE OR BUILD_DESKTOP OR BUILD_PLUGINS)
    set(KMIDI_REQUIRE_QT ON)
endif()

if(KMIDI_REQUIRE_QT)
    find_package(Qt6 COMPONENTS Core Widgets REQUIRED)
endif()

# JUCE setup
# Skip building juceaide to avoid macOS 15.0 compatibility issues
# We only need juce_dsp for penta_core, not the full JUCE build tools
# Requires a full JUCE clone (incl. extras/Build/CMake) in external/JUCE.
if(USE_KMI_DI_FINAL AND KMI_DI_FINAL_AVAILABLE AND EXISTS ${KMI_DI_FINAL_ROOT}/build/external/JUCE)
    message(STATUS "Using JUCE from KmiDi_FINAL")
    add_subdirectory(${KMI_DI_FINAL_ROOT}/build/external/JUCE JUCE)
elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/external/JUCE/CMakeLists.txt)
    message(STATUS "Using local JUCE (external/JUCE)")
    set(JUCE_DISABLE_JUCEAIDE_BUILD ON CACHE BOOL "Skip juceaide build")
    add_subdirectory(external/JUCE EXCLUDE_FROM_ALL)
else()
    message(FATAL_ERROR "JUCE not found. Clone the full JUCE repo (with root CMakeLists.txt and extras/Build/CMake/) into external/JUCE, e.g.: git clone https://github.com/juce-framework/JUCE external/JUCE")
endif()

# Readerwriterqueue (lock-free queue) for RT-safe OSC
# Note: This is a header-only library with no CMakeLists.txt, so we must
# explicitly create an INTERFACE target and ensure SOURCE_DIR is populated.
if(KMIDI_OFFLINE_BUILD)
    message(FATAL_ERROR "KMIDI_OFFLINE_BUILD=ON forbids FetchContent network pulls. Vendor readerwriterqueue, RTNeural, and googletest in external/ and use add_subdirectory or target_include_directories.")
endif()
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
# KmiDi engine (C ABI for RT harness and plugin/FFI)
# ============================================================================
add_subdirectory(engine)

# ============================================================================
# RT Harness (headless callback runner, P50/P90/P99 stats)
# ============================================================================
if(BUILD_RT_HARNESS)
  add_subdirectory(rt_harness)
endif()

# ============================================================================
# Penta-Core Library (C++ RT engines)
# ============================================================================
add_subdirectory(src_penta-core)

# ============================================================================
# Python Bindings (Optional)
# ============================================================================
option(BUILD_PYTHON_BINDINGS "Build Python bindings for penta-core" ON)

if(BUILD_PYTHON_BINDINGS)
    # Find pybind11
    find_package(pybind11 QUIET)
    if(pybind11_FOUND)
        message(STATUS "pybind11 found, building Python bindings")
        add_subdirectory(bindings)
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
            GIT_TAG 1fb1f075a5d66e85bfc8f488c3f3626840cb3a1d
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
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.mm
)

# Exclude app entry point and plugin sources from the core lib
list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/legacy/ui/qt_gui/main.cpp$")
list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/plugin/")
list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/engine/test_kelly\\.cpp$")
# Exclude Python bridge and cpp_music_brain Python bindings when pybind11 not available
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/bridge/kelly_bridge\\.cpp$")
    list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/cpp_music_brain/python/.*\\.cpp$")
    list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/python/.*\\.cpp$")
endif()
# NOTE: Previous line 31 had "list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/")"
# which would exclude ALL source files (since all paths contain "/src/"), making
# KELLY_CORE_SOURCES empty. This has been removed. If macro conflicts exist,
# they should be fixed by excluding specific problematic files/directories rather
# than excluding everything.

if(BUILD_KELLY_CORE)
add_library(KellyCore STATIC ${KELLY_CORE_SOURCES})

# Ensure ObjC++ file (for biometrics) is treated correctly if present
set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/biometric/BiometricInput.mm
    PROPERTIES COMPILE_LANGUAGES OBJCXX
)

target_include_directories(KellyCore PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/bridge
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(KellyCore PUBLIC
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

if(KMIDI_REQUIRE_QT)
    target_link_libraries(KellyCore PUBLIC
        Qt6::Core
        Qt6::Widgets
    )
endif()

if(pybind11_FOUND)
    target_link_libraries(KellyCore PUBLIC pybind11::pybind11)
endif()

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
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic
        -Wno-deprecated-declarations
        -Wno-unused-parameter
        -Wno-unused-variable
        -Wno-unused-private-field
        -Wno-unused-but-set-variable
        -Wno-unused-function
        -Wno-unused-const-variable
        -Wno-unused-lambda-capture
        -Wno-sign-compare
        -Wno-reorder
        -Wno-overloaded-virtual
        -Wno-switch>
)
# Wno-*: full warning-clean build; many params/vars are interface or reserved. Tidy per-file when touching.
# -Wno-deprecated-declarations: JUCE Font(height, style) → Font(FontOptions); migrate when editing UI.

# Kelly GUI Application (Legacy Qt - see legacy/ui/qt_gui/)
if(BUILD_DESKTOP)
    if(NOT BUILD_KELLY_CORE)
        message(FATAL_ERROR "BUILD_DESKTOP requires BUILD_KELLY_CORE=ON")
    endif()
    add_executable(KellyApp
        legacy/ui/qt_gui/main.cpp
        legacy/ui/qt_gui/main_window.cpp
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
    if(NOT BUILD_KELLY_CORE)
        message(FATAL_ERROR "BUILD_PLUGINS requires BUILD_KELLY_CORE=ON")
    endif()
    set(KMIDI_PLUGIN_ICON_BIG KmiDi_plugin_256.png)
    set(KMIDI_PLUGIN_ICON_SMALL KmiDi_plugin_64.png)
    set(KMIDI_ICON_ROOT ${CMAKE_SOURCE_DIR}/KmiDi_FINAL/assets/icons/plugin)
    set(KellyPlugin_ICON_ARGS "")
    if(EXISTS ${KMIDI_ICON_ROOT}/${KMIDI_PLUGIN_ICON_BIG} AND EXISTS ${KMIDI_ICON_ROOT}/${KMIDI_PLUGIN_ICON_SMALL})
        set(KellyPlugin_ICON_ARGS
            ICON_BIG ${KMIDI_ICON_ROOT}/${KMIDI_PLUGIN_ICON_BIG}
            ICON_SMALL ${KMIDI_ICON_ROOT}/${KMIDI_PLUGIN_ICON_SMALL}
        )
    else()
        message(WARNING "KmiDi plugin icons not found at ${KMIDI_ICON_ROOT}; building without plugin icons.")
    endif()
    # VST3/CLAP plugin. Note: current JUCE (external/JUCE) does not include CLAP in its
    # CMake plugin kinds, so only KellyPlugin_VST3 (and KellyPlugin_All) are created.
    juce_add_plugin(KellyPlugin
        COMPANY_NAME "Kelly"
        PLUGIN_MANUFACTURER_CODE Klly
        PLUGIN_CODE Klp1
        FORMATS VST3 CLAP
        PRODUCT_NAME "Kelly Emotion Processor"
        VST3_CATEGORIES Fx Synth
        CLAP_ID com.kelly.emotion-processor
        ${KellyPlugin_ICON_ARGS}
    )

    target_sources(KellyPlugin PRIVATE
        src/plugin/PluginProcessor.cpp
        src/plugin/PluginEditor.cpp
        src/plugin/PluginState.cpp
        src/plugin/MasterEQProcessor.cpp
    )

    target_link_libraries(KellyPlugin PRIVATE
        KellyCore
        juce::juce_audio_plugin_client
    )
    # VST3-only build: no VST2 SDK available, so disable VST2 state replacement code.
    target_compile_definitions(KellyPlugin PRIVATE
        JUCE_VST3_CAN_REPLACE_VST2=0
        JUCE_IGNORE_VST3_MISMATCHED_PARAMETER_ID_WARNING=1
    )
    if(TARGET KellyPlugin_VST3)
        target_compile_definitions(KellyPlugin_VST3 PRIVATE
            JUCE_VST3_CAN_REPLACE_VST2=0
            JUCE_IGNORE_VST3_MISMATCHED_PARAMETER_ID_WARNING=1
        )
    endif()
endif()

endif() # BUILD_KELLY_CORE

# ============================================================================
# Native macOS App from KmiDi_FINAL
# ============================================================================
if(BUILD_NATIVE_MACOS_APP AND KMI_DI_FINAL_AVAILABLE AND EXISTS ${KMI_DI_FINAL_ROOT}/apps/macOS)
    message(STATUS "Adding native macOS app build target from KmiDi_FINAL")

    # Custom target for native app build
    add_custom_target(native_macos_app
        COMMAND ${KMI_DI_FINAL_ROOT}/apps/macOS/build_macos_app.sh
        WORKING_DIRECTORY ${KMI_DI_FINAL_ROOT}/apps/macOS
        COMMENT "Building native macOS app from KmiDi_FINAL"
        VERBATIM
    )

    # Make it depend on DSP core if available
    if(KMI_DI_HAS_PURE_DSP)
        add_dependencies(native_macos_app kmidi_dsp_core)
    endif()
endif()

# ============================================================================
# Kelly FFI Library (for Tauri integration)
# ============================================================================
option(BUILD_KELLY_FFI "Build Kelly FFI library for external language bindings" ON)

if(BUILD_KELLY_FFI AND BUILD_KELLY_CORE)
    add_library(KellyFFI SHARED
        src/bridge/kelly_ffi.cpp
    )
    
    target_include_directories(KellyFFI PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    
    # Link JUCE and Qt PRIVATE so consumers (e.g. KellyFFIBenchmark) do not get a
    # second copy of JUCE. Two copies cause allocator mismatch: dylib's static
    # init (e.g. JuceVersionPrinter) allocates with dylib's JUCE heap, then exe's
    # JUCE can run release() and abort with "pointer being freed was not allocated".
    target_link_libraries(KellyFFI PUBLIC KellyCore PRIVATE
        Qt6::Core
        Qt6::Widgets
        juce::juce_audio_basics
        juce::juce_core
    )
    
    # Avoid JUCE debug static initializer (JuceVersionPrinter) in the dylib to
    # prevent any cross-boundary string ops during load.
    target_compile_definitions(KellyFFI PRIVATE
        KELLY_FFI_BUILDING_DLL
        JUCE_DISABLE_JUCE_VERSION_PRINTING=1
    )
    set_target_properties(KellyFFI PROPERTIES
        C_VISIBILITY_PRESET default
        CXX_VISIBILITY_PRESET default
        VISIBILITY_INLINES_HIDDEN FALSE
        VERSION 1.0.0
        SOVERSION 1
    )
    
    # Platform-specific settings
    if(APPLE)
        set_target_properties(KellyFFI PROPERTIES
            MACOSX_RPATH TRUE
            INSTALL_NAME_DIR "@rpath"
        )
    endif()
    
    # Copy FFI library to Tauri resources directory after build
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src-tauri)
        add_custom_command(TARGET KellyFFI POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory
                ${CMAKE_CURRENT_SOURCE_DIR}/src-tauri/resources
            COMMAND ${CMAKE_COMMAND} -E copy
                $<TARGET_FILE:KellyFFI>
                ${CMAKE_CURRENT_SOURCE_DIR}/src-tauri/resources/
            COMMENT "Copying KellyFFI library to Tauri resources"
        )
    endif()
    
    # Install FFI library
    install(TARGETS KellyFFI
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
    
    # Install FFI header
    install(FILES src/bridge/kelly_ffi.h
        DESTINATION include/kelly
    )
    
    # Performance benchmark for FFI integration
    add_executable(KellyFFIBenchmark
        tests/performance/benchmark_integration.cpp
    )
    
    target_link_libraries(KellyFFIBenchmark PRIVATE
        KellyFFI
    )
    
    target_include_directories(KellyFFIBenchmark PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    
    message(STATUS "Kelly FFI library will be built")
else()
    message(STATUS "Kelly FFI library disabled (requires BUILD_KELLY_CORE=ON)")
endif()

# Tests
if(BUILD_TESTS AND BUILD_KELLY_CORE)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/external/Catch2/CMakeLists.txt)
        enable_testing()
        add_subdirectory(external/Catch2 EXCLUDE_FROM_ALL)
        add_executable(KellyTests
            tests/cpp/test_emotion_engine.cpp
            tests/cpp/test_midi_pipeline.cpp
            tests/cpp/test_chord_diagnostics.cpp
            tests/cpp/test_ml_pipeline.cpp
        )
        target_link_libraries(KellyTests PRIVATE
            KellyCore
            Catch2::Catch2WithMain
        )
        include(CTest)
        include(Catch2)
        catch_discover_tests(KellyTests)
    else()
        message(WARNING "Catch2 not found; skipping tests. Set BUILD_TESTS=OFF or add external/Catch2.")
    endif()
endif()

# DSP core testing (if using KmiDi_FINAL)
if(KMI_DI_HAS_PURE_DSP)
    message(STATUS "Adding DSP core test target")

    add_executable(dsp_core_test
        ${KMI_DI_FINAL_ROOT}/engine/src/dsp/test/test_dsp_core.cpp
    )
    target_link_libraries(dsp_core_test PRIVATE kmidi_dsp_core)
    target_include_directories(dsp_core_test PRIVATE ${KMI_DI_FINAL_ROOT}/engine/include)

    # Add test
    add_test(NAME DSP_Core_Test COMMAND dsp_core_test)
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
if(BUILD_KELLY_CORE)
    set(KELLY_INSTALL_TARGETS KellyCore)
    if(BUILD_DESKTOP)
        list(APPEND KELLY_INSTALL_TARGETS KellyApp)
    endif()

    install(TARGETS ${KELLY_INSTALL_TARGETS}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )

    install(DIRECTORY src/
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
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/python/penta_core)
        install(DIRECTORY python/penta_core
            DESTINATION share/kelly/python
            FILES_MATCHING PATTERN "*.py"
        )
    endif()
endif()
