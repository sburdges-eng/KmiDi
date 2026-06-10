cmake_minimum_required(VERSION 3.27)
project(Kelly VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

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
# MIDI-CI Property Exchange daemon (JSON microformat -> SysEx via libremidi)
option(BUILD_MIDI_CI_DAEMON "Build MIDI-CI microformat-to-SysEx daemon" OFF)
# Sanitizers (ASan+UBSan, TSan, or RTSan; Debug builds only; mutually exclusive)
option(KMIDI_ENABLE_ASAN "Enable AddressSanitizer + UndefinedBehaviorSanitizer" OFF)
option(KMIDI_ENABLE_TSAN "Enable ThreadSanitizer (mutually exclusive with ASAN)" OFF)
option(KMIDI_ENABLE_RTSAN "Enable LLVM RealtimeSanitizer (mutually exclusive with ASAN/TSAN; requires Clang 20+)" OFF)
# Rust Intent IR staticlib. OFF exists for TSan certification builds: the
# prebuilt archive leaks compiler_builtins symbols (__divdc3 etc.), so the
# linker drags in Rust std and rust_eh_personality, overflowing the arm64
# compact-unwind personality limit once TSan's runtime joins the link.
# FFI builds require this ON (KellyFFI embeds the staticlib).
option(KMIDI_ENABLE_INTENT_IR "Link the Rust Intent IR staticlib into KellyCore" ON)

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

# Qt is only required when Qt-backed targets are being built. KellyCore and
# the JUCE plugins include no Qt headers — the only Qt consumer is the legacy
# desktop app (legacy/ui/qt_gui/, gated by KMIDI_BUILD_QT_UI) — so a missing
# Qt6 must not fail configure for core/test/CI builds.
set(KMIDI_REQUIRE_QT OFF)
if(KMIDI_BUILD_QT_UI OR BUILD_DESKTOP)
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

# ConcurrentQueue (MPMC lock-free queue) for StateBridge
# Vendored at third_party/concurrentqueue/ (MIT license).
# Used by StateBridge whose emitStateUpdate is called from multiple engine
# threads (MPSC pattern) — ReaderWriterQueue is SPSC and violated that contract.
add_library(concurrentqueue INTERFACE)
target_include_directories(concurrentqueue INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/concurrentqueue
)

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
# Intent IR Rust crate (validation, clamping, builder)
# ============================================================================
if(KMIDI_ENABLE_INTENT_IR AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/engine/intent_ir/CMakeLists.txt)
    add_subdirectory(engine/intent_ir)
    set(KMIDI_HAS_INTENT_IR ON)
    message(STATUS "Intent IR Rust crate enabled")
elseif(NOT KMIDI_ENABLE_INTENT_IR)
    message(STATUS "Intent IR Rust crate disabled (KMIDI_ENABLE_INTENT_IR=OFF)")
endif()

# ============================================================================
# In-repo DSP core (merged from KmiDi_FINAL; optional link into KellyCore)
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/engine/src/dsp/CMakeLists.txt)
    add_subdirectory(engine/src/dsp)
    set(KMIDI_HAS_INREPO_DSP ON)
endif()

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
# daiw_core — standalone RT-safe primitives library
# ============================================================================
add_subdirectory(libs/daiw)

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
# intent_ir_ffi.cpp references Rust FFI symbols unconditionally; without the
# staticlib those references would dangle in libKellyCore.a and any consumer
# that pulls the object fails at link. Keep the object out of the archive
# entirely when Intent IR is disabled.
if(NOT KMIDI_ENABLE_INTENT_IR)
    list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/bridge/intent_ir_ffi\\.cpp$")
endif()
# Exclude daiw-namespace files (built separately as daiw_core static lib)
list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/dsp/audio_buffer\\.cpp$")
list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/dsp/simd_ops\\.cpp$")
list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/dsp/dsp\\.cpp$")
list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/core/memory\\.cpp$")
list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/core/logging\\.cpp$")
list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/core/types\\.cpp$")
list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/midi/midi_engine\\.cpp$")
list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/midi/MidiIO\\.cpp$")
list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/midi/MidiMessage\\.cpp$")
list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/midi/MidiSequence\\.cpp$")
list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/harmony/chord\\.cpp$")
list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/harmony/progression\\.cpp$")
list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/audio/AudioFile\\.cpp$")
list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/export/StemExporter\\.cpp$")
list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/project/ProjectFile\\.cpp$")
list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/python/groove_bindings\\.cpp$")

# Intra-library ODR fixes 2026-04-14 (caught by scripts/audit/intra_lib_odr_scan.py):
#   - src/core/emotion_thesaurus.cpp is a self-labeled legacy stub from
#     'early architecture iterations' that returns nullptr/no-op for
#     everything. It defines the same kelly::EmotionThesaurus symbols
#     as the real 216-node implementation in src/engine/EmotionThesaurus.cpp.
#     Linker silently picks one; if it picks the stub, the engine returns
#     no-ops in production. Exclude the stub so the real impl wins.
list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/core/emotion_thesaurus\\.cpp$")
#   - src/ui/MidiKompanionLookAndFeel.cpp is a byte-identical copy of
#     src/ui/KellyLookAndFeel.cpp (cmp -s clean). Both define the same
#     kelly::KellyLookAndFeel virtual table + thunks. Exclude the copy.
list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/ui/MidiKompanionLookAndFeel\\.cpp$")

# BiometricInput: .cpp and .mm define overlapping symbols.  Select exactly
# one per platform so the linker doesn't see duplicates.
if(APPLE)
    list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/biometric/BiometricInput\\.cpp$")
else()
    list(FILTER KELLY_CORE_SOURCES EXCLUDE REGEX "/src/biometric/BiometricInput\\.mm$")
endif()

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
    daiw_core
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
    concurrentqueue
    penta_core
)
# Link Rust Intent IR if available
if(KMIDI_HAS_INTENT_IR)
    target_link_libraries(KellyCore PUBLIC ${INTENT_IR_RUST_ARCHIVE})
    add_dependencies(KellyCore intent_ir_rust_lib)
    target_compile_definitions(KellyCore PRIVATE INTENT_IR_RUST_AVAILABLE)
    message(STATUS "KellyCore: Intent IR Rust linkage enabled (${INTENT_IR_RUST_ARCHIVE})")
endif()

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

# Python bridge activation — enables 52 call sites across 9 bridge files
if(pybind11_FOUND)
    target_compile_definitions(KellyCore PRIVATE PYTHON_AVAILABLE)
    message(STATUS "KellyCore: PYTHON_AVAILABLE defined (bridge layer active)")
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

# Sanitizer instrumentation (Debug builds; compile after all KellyCore config is settled)
# ASan, TSan, and RTSan share runtime symbols and are mutually exclusive.
set(_KMIDI_SAN_COUNT 0)
foreach(_san KMIDI_ENABLE_ASAN KMIDI_ENABLE_TSAN KMIDI_ENABLE_RTSAN)
    if(${_san})
        math(EXPR _KMIDI_SAN_COUNT "${_KMIDI_SAN_COUNT} + 1")
    endif()
endforeach()
if(_KMIDI_SAN_COUNT GREATER 1)
    message(FATAL_ERROR
        "KMIDI_ENABLE_ASAN, KMIDI_ENABLE_TSAN, and KMIDI_ENABLE_RTSAN are mutually exclusive (pick one)")
endif()
if(KMIDI_ENABLE_ASAN)
    include(Sanitizers)
    enable_sanitizers(KellyCore)
    message(STATUS "KellyCore: ASan + UBSan enabled")
endif()
if(KMIDI_ENABLE_TSAN)
    include(Sanitizers)
    enable_thread_sanitizer(KellyCore)
    message(STATUS "KellyCore: ThreadSanitizer enabled")
endif()
if(KMIDI_ENABLE_RTSAN)
    include(Sanitizers)
    enable_realtime_sanitizer(KellyCore)
    message(STATUS "KellyCore: RealtimeSanitizer enabled — processBlock must be [[clang::nonblocking]]")
endif()

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

    # KellyCore's PythonBridgeBase.cpp / StateBridge.cpp call the CPython
    # C API directly. Plugin variants (KellyPlugin_VST3, KellyPlugin_AU,
    # KellyPlugin_Standalone, KellyPlugin_CLAP) all link KellyCore.a in
    # their final link step, so each one needs libpython on its link line
    # — otherwise the static linker reports _Py_Initialize / __Py_Dealloc
    # as undefined. find_package is repeated here (it's also inside the
    # BUILD_KELLY_FFI block) because the Python3::Python imported target
    # is scope-local.
    find_package(Python3 COMPONENTS Development QUIET)

    # VST3/CLAP plugin. Note: current JUCE (external/JUCE) does not include CLAP in its
    # CMake plugin kinds, so only KellyPlugin_VST3 (and KellyPlugin_All) are created.
    juce_add_plugin(KellyPlugin
        COMPANY_NAME "Kelly"
        BUNDLE_ID com.kelly.emotion-processor
        PLUGIN_MANUFACTURER_CODE Klly
        PLUGIN_CODE Klp1
        FORMATS AU VST3 CLAP Standalone
        PRODUCT_NAME "Kelly Emotion Processor"
        AU_MAIN_TYPE kAudioUnitType_MusicEffect
        VST3_CATEGORIES Fx Synth
        CLAP_ID com.kelly.emotion-processor

        # ---------------------------------------------------------------------
        # macOS 14+ permissions for the Standalone target.
        # JUCE emits matching Info.plist keys + entitlements. Without these the
        # standalone app silently fails to accept audio input, open BLE-MIDI
        # peripherals, bind to local-network peers (virtual MIDI session /
        # mDNS / RTP-MIDI / localhost HTTP), or drive Logic via the Apple
        # Events fallback. Only the Standalone wrapper prompts the user;
        # VST3/AU/CLAP inherit the host's entitlements so these keys are no-ops
        # for the plugin formats.
        # ---------------------------------------------------------------------
        MICROPHONE_PERMISSION_ENABLED TRUE
        MICROPHONE_PERMISSION_TEXT
            "Kelly Emotion Processor uses the microphone to analyze audio input for emotion and pitch tracking."

        BLUETOOTH_PERMISSION_ENABLED TRUE
        BLUETOOTH_PERMISSION_TEXT
            "Kelly Emotion Processor uses Bluetooth to connect to BLE-MIDI controllers and DAW surfaces."

        # AppleScript send-events is required for the Logic Pro bridge's
        # structural-command fallback path (track create/rename/routing —
        # things Mackie Control Universal cannot express). Drop this flag
        # if the bridge is ever restricted to MCU-only.
        SEND_APPLE_EVENTS_PERMISSION_ENABLED TRUE
        SEND_APPLE_EVENTS_PERMISSION_TEXT
            "Kelly Emotion Processor sends Apple Events to Logic Pro for track management operations the Mackie Control protocol cannot perform."

        # macOS 14 introduced explicit Local Network consent. This covers:
        #   * virtual MIDI peers on the local host,
        #   * RTP-MIDI (Network MIDI Setup) sessions advertised via Bonjour,
        #   * the Tauri bridge connecting to the FastAPI backend on 127.0.0.1:8000,
        #   * any mDNS/Bonjour service discovery.
        NETWORK_MULTICAST_PERMISSION_ENABLED TRUE
        NETWORK_MULTICAST_PERMISSION_TEXT
            "Kelly Emotion Processor uses the local network to discover MIDI peers, DAW control surfaces, and the companion audio engine."

        # Sandbox OFF for the Standalone: plugin hosts already run us sandboxed
        # when formatted AU/VST3/CLAP, and the standalone needs open access to
        # the user's model/dataset folders (KELLY_MODEL_ROOT, KELLY_AUDIO_DATA_ROOT)
        # which are not stable across App Store sandbox containers. Notarization
        # still applies for distribution.
        APP_SANDBOX_ENABLED FALSE
        HARDENED_RUNTIME_ENABLED TRUE

        # Declare the Bonjour service types the app is allowed to browse on
        # the local network. `_apple-midi._udp` is RTP-MIDI (Network MIDI
        # Setup sessions); `_kmidi-bridge._tcp` is our own service for the
        # Tauri <-> KellyFFI companion discovery.
        PLIST_TO_MERGE "<dict>
            <key>NSBonjourServices</key>
            <array>
                <string>_apple-midi._udp</string>
                <string>_kmidi-bridge._tcp</string>
            </array>
        </dict>"

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
        # juce_audio_utils + juce_audio_devices are required by the JUCE
        # Standalone wrapper (juce_audio_plugin_client_Standalone.cpp #errors
        # without them). The AU/VST3/CLAP formats build fine without these,
        # but linking them is a no-op for the plugin formats.
        juce::juce_audio_utils
        juce::juce_audio_devices
        # KellyCore embeds PythonBridgeBase.cpp / StateBridge.cpp which use
        # the CPython C API. Plugin variants must link libpython at link
        # time or the linker reports unresolved _Py_Initialize / __Py_Dealloc
        # symbols. Matches the same pattern used for KellyFFI above.
        $<$<TARGET_EXISTS:Python3::Python>:Python3::Python>
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

    # Propagate libpython to every plugin variant juce_add_plugin produces.
    # PRIVATE on KellyPlugin (the SharedCode static lib) does not transit
    # through to the variants' final link step, so we apply it directly.
    foreach(_kelly_plugin_variant
            KellyPlugin_VST3 KellyPlugin_AU KellyPlugin_AUv3
            KellyPlugin_CLAP KellyPlugin_LV2 KellyPlugin_Standalone
            KellyPlugin_Unity KellyPlugin_AAX KellyPlugin_VST)
        if(TARGET ${_kelly_plugin_variant} AND TARGET Python3::Python)
            target_link_libraries(${_kelly_plugin_variant} PRIVATE
                Python3::Python)
        endif()
    endforeach()
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
if(BUILD_KELLY_FFI AND BUILD_KELLY_CORE AND NOT KMIDI_ENABLE_INTENT_IR)
    message(FATAL_ERROR "BUILD_KELLY_FFI requires KMIDI_ENABLE_INTENT_IR=ON (KellyFFI embeds the Rust intent_ir staticlib)")
endif()

if(BUILD_KELLY_FFI AND BUILD_KELLY_CORE)
    add_library(KellyFFI SHARED
        src/bridge/kelly_ffi.cpp
    )
    
    target_include_directories(KellyFFI PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    
    # Link KellyCore, JUCE and Qt PRIVATE so consumers (e.g. KellyFFIBenchmark,
    # Python bindings, Tauri) do not transitively inherit KellyCore's PUBLIC JUCE
    # linkage. Two copies of JUCE cause allocator mismatch: dylib's static init
    # (e.g. JuceVersionPrinter) allocates with dylib's JUCE heap, then exe's JUCE
    # can run release() and abort with "pointer being freed was not allocated".
    # Consumers of KellyFFI must use the C ABI in kelly_ffi.h, not KellyCore symbols.
    # Python3::Python: KellyCore embeds PythonBridgeBase.cpp / StateBridge.cpp
    # which call the Python C API (_Py_Initialize, _PyUnicode_FromString,
    # __Py_Dealloc, etc.). The KellyFFI dylib must link libpython explicitly
    # so those symbols resolve at link time on macOS. find_package(Python3)
    # is already invoked by bindings/CMakeLists.txt; we re-find it here for
    # the top-level scope. PRIVATE keeps it out of the C ABI consumer surface.
    find_package(Python3 COMPONENTS Development QUIET)
    target_link_libraries(KellyFFI PRIVATE KellyCore
        juce::juce_audio_basics
        juce::juce_core
        $<$<TARGET_EXISTS:Python3::Python>:Python3::Python>
    )
    # Qt is linked only when Qt-backed targets are enabled; no KellyFFI or
    # KellyCore source includes Qt headers, so this stays PRIVATE-and-optional.
    if(KMIDI_REQUIRE_QT)
        target_link_libraries(KellyFFI PRIVATE Qt6::Core Qt6::Widgets)
    endif()
    
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

# MIDI-CI daemon (optional; requires network for FetchContent unless vendored)
if(BUILD_MIDI_CI_DAEMON AND NOT KMIDI_OFFLINE_BUILD)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tools/midi_ci_daemon/CMakeLists.txt)
        add_subdirectory(tools/midi_ci_daemon)
        message(STATUS "MIDI-CI daemon enabled (tools/midi_ci_daemon)")
    else()
        message(WARNING "BUILD_MIDI_CI_DAEMON=ON but tools/midi_ci_daemon/CMakeLists.txt not found")
    endif()
endif()

# Tests
if(BUILD_TESTS AND BUILD_KELLY_CORE)
    # Register tests with CTest unconditionally. The standalone RT/DSP tests
    # below are designed to run without Catch2 (and even without JUCE), so
    # enable_testing() must run before their add_test() calls regardless of
    # whether the optional Catch2-based tests are present.
    enable_testing()
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/external/Catch2/CMakeLists.txt)
        add_subdirectory(external/Catch2 EXCLUDE_FROM_ALL)
        include(CTest)
        include(Catch2)
        # The previous `KellyTests` target referenced four test files that do
        # not exist in the tree (test_emotion_engine.cpp, test_midi_pipeline.cpp,
        # test_chord_diagnostics.cpp, test_ml_pipeline.cpp), silently breaking
        # `-DBUILD_TESTS=ON`.  Targets below are the tests that actually exist
        # and compile.  Add a replacement `KellyTests` target here only when
        # there are real source files to put in it.
        # AudioEmotionRunner tests
        add_executable(AudioEmotionRunnerTests
            tests/cpp/test_audio_emotion_runner.cpp
        )
        target_link_libraries(AudioEmotionRunnerTests PRIVATE
            KellyCore
            Catch2::Catch2WithMain
        )
        target_include_directories(AudioEmotionRunnerTests PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        catch_discover_tests(AudioEmotionRunnerTests)
        # EmotionSchema parity tests
        add_executable(EmotionSchemaTests
            tests/cpp/test_emotion_schema.cpp
        )
        target_link_libraries(EmotionSchemaTests PRIVATE
            KellyCore
            Catch2::Catch2WithMain
        )
        target_include_directories(EmotionSchemaTests PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        catch_discover_tests(EmotionSchemaTests)
    else()
        message(WARNING "Catch2 not found; skipping tests. Set BUILD_TESTS=OFF or add external/Catch2.")
    endif()

    # Runtime contract tests — Catch2 executable exercising the kelly_* C ABI
    # through KellyFFI ONLY. Executables linking KellyFFI must not also link
    # JUCE or KellyCore: KellyFFI links both PRIVATE, and a second JUCE copy
    # in the exe causes allocator mismatch (see KellyFFI target comment).
    # Catch2 comes from vendored external/Catch2 when present (block above),
    # otherwise it is fetched — mirroring the googletest FetchContent in the
    # BUILD_PENTA_TESTS block.
    if(TARGET KellyFFI)
        if(NOT TARGET Catch2::Catch2WithMain AND NOT KMIDI_OFFLINE_BUILD)
            include(FetchContent)
            FetchContent_Declare(
                Catch2
                GIT_REPOSITORY https://github.com/catchorg/Catch2.git
                GIT_TAG v3.5.2
                GIT_SHALLOW TRUE
            )
            FetchContent_MakeAvailable(Catch2)
        endif()
        if(TARGET Catch2::Catch2WithMain)
            add_executable(runtime_contract_tests
                tests/cpp/test_kelly_ffi.cpp
            )
            target_compile_features(runtime_contract_tests PRIVATE cxx_std_20)
            find_package(Threads REQUIRED)
            target_link_libraries(runtime_contract_tests PRIVATE
                KellyFFI
                Catch2::Catch2WithMain
                Threads::Threads
            )
            if(COMMAND add_test)
                add_test(NAME runtime_contract_tests COMMAND runtime_contract_tests)
            endif()
        else()
            message(WARNING "runtime_contract_tests skipped: Catch2 unavailable (vendor external/Catch2 or allow FetchContent)")
        endif()
    endif()

    # RTState seqlock torture test — no Catch2, no KellyCore, no JUCE.
    # Standalone executable that exercises the header-only seqlock protocol
    # with a writer/reader thread pair over 1M iterations. Runnable even in
    # worktrees where external/JUCE is absent.
    add_executable(RTStateSeqlockTest
        tests/cpp/test_rt_state_seqlock.cpp
    )
    target_include_directories(RTStateSeqlockTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_compile_features(RTStateSeqlockTest PRIVATE cxx_std_20)
    if(CMAKE_THREADS_FOUND OR TRUE)
        find_package(Threads REQUIRED)
        target_link_libraries(RTStateSeqlockTest PRIVATE Threads::Threads)
    endif()
    if(COMMAND add_test)
        add_test(NAME RTStateSeqlockTest COMMAND RTStateSeqlockTest)
    endif()

    # RT event scheduler test — no Catch2, no KellyCore, no JUCE.
    # Verifies sample-accurate dispatch ordering, cross-block scheduling,
    # transport-jump AllNotesOff, transaction commit/abort, rollback via
    # cancel events, and a 4-producer / 1-consumer stress smoke test.
    add_executable(RTEventSchedulerTest
        tests/cpp/test_rt_event_scheduler.cpp
    )
    target_include_directories(RTEventSchedulerTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_compile_features(RTEventSchedulerTest PRIVATE cxx_std_20)
    find_package(Threads REQUIRED)
    target_link_libraries(RTEventSchedulerTest PRIVATE Threads::Threads)
    if(COMMAND add_test)
        add_test(NAME RTEventSchedulerTest COMMAND RTEventSchedulerTest)
    endif()

    # DSP graph DAG test — no Catch2, no KellyCore, no JUCE.
    # Verifies topological sort, fan-out buffer sharing, cycle rejection,
    # capacity guards, and oversize-block safety on daiw::rt::DSPGraph.
    add_executable(DSPGraphTest
        tests/cpp/test_dsp_graph.cpp
    )
    target_include_directories(DSPGraphTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_compile_features(DSPGraphTest PRIVATE cxx_std_20)
    if(COMMAND add_test)
        add_test(NAME DSPGraphTest COMMAND DSPGraphTest)
    endif()

    # GPU scheduler test — no Catch2, no KellyCore, no JUCE.
    # Verifies CpuGpuScheduler runs submitted jobs, RT polling pattern,
    # MetalGpuScheduler stub reports correctly, and select_default picks
    # the right backend.
    add_executable(GpuSchedulerTest
        tests/cpp/test_gpu_scheduler.cpp
    )
    target_include_directories(GpuSchedulerTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_compile_features(GpuSchedulerTest PRIVATE cxx_std_20)
    target_link_libraries(GpuSchedulerTest PRIVATE Threads::Threads)
    if(COMMAND add_test)
        add_test(NAME GpuSchedulerTest COMMAND GpuSchedulerTest)
    endif()

    # Cross-process shared-memory IPC test — POSIX only.
    # fork()s a child that attaches to a parent-created SHM region. Verifies
    # bidirectional MPSC command + event rings and RTState mirror.
    if(UNIX)
        add_executable(SharedIpcTest
            tests/cpp/test_shared_ipc.cpp
        )
        target_include_directories(SharedIpcTest PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
        target_compile_features(SharedIpcTest PRIVATE cxx_std_20)
        target_link_libraries(SharedIpcTest PRIVATE Threads::Threads)
        if(COMMAND add_test)
            add_test(NAME SharedIpcTest COMMAND SharedIpcTest)
        endif()
    endif()

    # SIMD kernels parity test — no Catch2, no KellyCore, no JUCE.
    # Verifies that penta::simd free-function primitives (dot_product_f32,
    # multiply_add_f32, max_element_f32) agree with a scalar reference to
    # within 1e-5 for edge sizes {0,1,3,4,5,7,8,9,15,16,17,31,32,1024}.
    # On Apple Silicon the NEON path fires; on x86-64 with -mavx2, the AVX2
    # path fires; otherwise the portable scalar fallback is tested.
    add_executable(SIMDKernelsTest
        tests/cpp/test_simd_kernels.cpp
    )
    target_include_directories(SIMDKernelsTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_compile_features(SIMDKernelsTest PRIVATE cxx_std_20)
    if(COMMAND add_test)
        add_test(NAME SIMDKernelsTest COMMAND SIMDKernelsTest)
    endif()

    # MultiModelProcessor lifecycle/ownership test — standalone main, links
    # KellyCore (juce_core arrives transitively via KellyCore's PUBLIC links,
    # so the executable must not link JUCE modules directly).
    add_executable(MultiModelProcessorTest
        tests/cpp/test_multi_model_processor.cpp
    )
    target_link_libraries(MultiModelProcessorTest PRIVATE KellyCore)
    target_include_directories(MultiModelProcessorTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_features(MultiModelProcessorTest PRIVATE cxx_std_20)
    if(COMMAND add_test)
        add_test(NAME MultiModelProcessorTest COMMAND MultiModelProcessorTest)
        # Lifecycle torture must never wedge an overnight ctest run.
        set_tests_properties(MultiModelProcessorTest PROPERTIES TIMEOUT 120)
    endif()

    # Instrument the test executables with whichever sanitizer is active so the
    # ctest gate runs under the same ASan/UBSan/TSan/RTSan as the KellyCore
    # build. Guarded by TARGET so Catch2-only targets are skipped when absent.
    if(KMIDI_ENABLE_ASAN OR KMIDI_ENABLE_TSAN OR KMIDI_ENABLE_RTSAN)
        include(Sanitizers)
        foreach(_kmidi_test_tgt
                AudioEmotionRunnerTests EmotionSchemaTests
                runtime_contract_tests
                RTStateSeqlockTest RTEventSchedulerTest DSPGraphTest
                GpuSchedulerTest SharedIpcTest SIMDKernelsTest
                MultiModelProcessorTest)
            if(TARGET ${_kmidi_test_tgt})
                if(KMIDI_ENABLE_ASAN)
                    enable_sanitizers(${_kmidi_test_tgt})
                elseif(KMIDI_ENABLE_TSAN)
                    enable_thread_sanitizer(${_kmidi_test_tgt})
                else()
                    enable_realtime_sanitizer(${_kmidi_test_tgt})
                endif()
            endif()
        endforeach()
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
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/penta_core/CMakeLists.txt)
        add_subdirectory(tests/penta_core)
    else()
        message(STATUS
            "tests/penta_core/ has no standalone CMakeLists.txt; the "
            "penta_tests target is built from tests/penta_core/*.cpp by "
            "src_penta-core/CMakeLists.txt instead.")
    endif()
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
