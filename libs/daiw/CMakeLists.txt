# daiw_core — RT-safe music primitives (standalone internal library)
#
# The daiw namespace provides low-level types and utilities used by
# both penta (RT engines) and kelly (intelligence layer).
# Dependency chain: daiw → penta → kelly
#
# Source files remain in their original directories under src/ but are
# excluded from KellyCore's GLOB_RECURSE and compiled here instead.

add_library(daiw_core STATIC
    # DSP primitives
    ${CMAKE_SOURCE_DIR}/src/dsp/audio_buffer.cpp
    ${CMAKE_SOURCE_DIR}/src/dsp/simd_ops.cpp
    ${CMAKE_SOURCE_DIR}/src/dsp/dsp.cpp
    # Core utilities
    ${CMAKE_SOURCE_DIR}/src/core/memory.cpp
    ${CMAKE_SOURCE_DIR}/src/core/logging.cpp
    ${CMAKE_SOURCE_DIR}/src/core/types.cpp
    # MIDI primitives
    ${CMAKE_SOURCE_DIR}/src/midi/midi_engine.cpp
    ${CMAKE_SOURCE_DIR}/src/midi/MidiIO.cpp
    ${CMAKE_SOURCE_DIR}/src/midi/MidiMessage.cpp
    ${CMAKE_SOURCE_DIR}/src/midi/MidiSequence.cpp
    # Harmony primitives
    ${CMAKE_SOURCE_DIR}/src/harmony/chord.cpp
    ${CMAKE_SOURCE_DIR}/src/harmony/progression.cpp
    # Audio I/O
    ${CMAKE_SOURCE_DIR}/src/audio/AudioFile.cpp
    # Export
    ${CMAKE_SOURCE_DIR}/src/export/StemExporter.cpp
    # Project
    ${CMAKE_SOURCE_DIR}/src/project/ProjectFile.cpp
)

target_include_directories(daiw_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# daiw uses JUCE for audio I/O, file handling, and DSP
target_link_libraries(daiw_core PUBLIC
    juce::juce_core
    juce::juce_audio_basics
    juce::juce_audio_formats
)

set_target_properties(daiw_core PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)
