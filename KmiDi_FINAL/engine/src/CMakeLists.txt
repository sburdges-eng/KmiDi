# Core C++ library
cmake_minimum_required(VERSION 3.20)
project(KmiDiCore LANGUAGES CXX)

set(PENTA_CORE_SOURCES
    # Common utilities
    common/RTMemoryPool.cpp
    common/RTLogger.cpp

    # Harmony engine
    harmony/ChordAnalyzer.cpp
    harmony/ChordAnalyzerSIMD.cpp
    harmony/ScaleDetector.cpp
    harmony/VoiceLeading.cpp
    harmony/HarmonyEngine.cpp

    # Groove analysis
    groove/OnsetDetector.cpp
    groove/TempoEstimator.cpp
    groove/RhythmQuantizer.cpp
    groove/GrooveEngine.cpp

    # Diagnostics
    diagnostics/PerformanceMonitor.cpp
    diagnostics/AudioAnalyzer.cpp
    diagnostics/DiagnosticsEngine.cpp

    # OSC communication
    osc/OSCServer.cpp
    osc/OSCClient.cpp
    osc/OSCMessage.cpp
    osc/OSCHub.cpp
    osc/RTMessageQueue.cpp
)

set(PENTA_CORE_HEADERS
    ${PROJECT_SOURCE_DIR}/include/penta/common/RTMemoryPool.h
    ${PROJECT_SOURCE_DIR}/include/penta/common/RTLogger.h
    ${PROJECT_SOURCE_DIR}/include/penta/common/RTTypes.h

    ${PROJECT_SOURCE_DIR}/include/penta/harmony/ChordAnalyzer.h
    ${PROJECT_SOURCE_DIR}/include/penta/harmony/ScaleDetector.h
    ${PROJECT_SOURCE_DIR}/include/penta/harmony/VoiceLeading.h
    ${PROJECT_SOURCE_DIR}/include/penta/harmony/HarmonyEngine.h

    ${PROJECT_SOURCE_DIR}/include/penta/groove/OnsetDetector.h
    ${PROJECT_SOURCE_DIR}/include/penta/groove/TempoEstimator.h
    ${PROJECT_SOURCE_DIR}/include/penta/groove/RhythmQuantizer.h
    ${PROJECT_SOURCE_DIR}/include/penta/groove/GrooveEngine.h

    ${PROJECT_SOURCE_DIR}/include/penta/diagnostics/PerformanceMonitor.h
    ${PROJECT_SOURCE_DIR}/include/penta/diagnostics/AudioAnalyzer.h
    ${PROJECT_SOURCE_DIR}/include/penta/diagnostics/DiagnosticsEngine.h

    ${PROJECT_SOURCE_DIR}/include/penta/osc/OSCServer.h
    ${PROJECT_SOURCE_DIR}/include/penta/osc/OSCClient.h
    ${PROJECT_SOURCE_DIR}/include/penta/osc/OSCMessage.h
    ${PROJECT_SOURCE_DIR}/include/penta/osc/RTMessageQueue.h
    ${PROJECT_SOURCE_DIR}/include/penta/osc/OSCHub.h
)

add_library(penta_core STATIC
    ${PENTA_CORE_SOURCES}
    ${PENTA_CORE_HEADERS}
)

set_target_properties(penta_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(penta_core PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(penta_core PUBLIC
    oscpack
    readerwriterqueue
)

# Enable AVX2 SIMD optimizations if available
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
if(COMPILER_SUPPORTS_AVX2)
    target_compile_options(penta_core PRIVATE -mavx2 -mpopcnt)
    message(STATUS "AVX2 SIMD optimizations enabled")
else()
    message(STATUS "AVX2 not available, using scalar fallback")
endif()

# Platform-specific threading
if(UNIX)
    target_link_libraries(penta_core PUBLIC pthread)
endif()

# Intent IR Rust crate
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../intent_ir intent_ir_build)

# Intent IR Adapter (C++ wrapper)
set(INTENT_IR_ADAPTER_SOURCES
    common/IntentIRAdapter.cpp
    ${PROJECT_SOURCE_DIR}/../shared/src/kmidi/IntentIR_impl.cpp
    ${PROJECT_SOURCE_DIR}/../shared/src/kmidi/IntentIR_JSON.cpp
)

set(INTENT_IR_ADAPTER_HEADERS
    common/IntentIRAdapter.h
    common/IntentIRExtractor.h
)

add_library(intent_ir_adapter STATIC
    ${INTENT_IR_ADAPTER_SOURCES}
    ${INTENT_IR_ADAPTER_HEADERS}
)

target_include_directories(intent_ir_adapter PUBLIC
    ${PROJECT_SOURCE_DIR}/../shared/include
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include  # For generated FFI header
)

target_link_libraries(intent_ir_adapter PUBLIC
    ${INTENT_IR_RUST_LIB}
)

# Link cJSON for JSON serialization (if available)
find_library(CJSON_LIB cjson)
if(CJSON_LIB)
    target_link_libraries(intent_ir_adapter PUBLIC ${CJSON_LIB})
    target_compile_definitions(intent_ir_adapter PRIVATE HAVE_CJSON)
endif()

add_dependencies(intent_ir_adapter intent_ir_rust_lib)

# Export for parent CMakeLists
set(INTENT_IR_ADAPTER_LIB intent_ir_adapter PARENT_SCOPE)

# Export intent_ir_adapter for use in main CMakeLists
set(INTENT_IR_ADAPTER_LIB intent_ir_adapter PARENT_SCOPE)
