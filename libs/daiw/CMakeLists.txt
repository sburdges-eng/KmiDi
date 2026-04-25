# daiw_core — RT-safe music primitives (interface-only after 2026 Q2 audit)
#
# The daiw namespace originally exposed low-level audio_buffer / simd_ops /
# midi_engine / chord / progression / AudioFile / etc. as separate .cpp
# translation units inside this static library. Per docs/AUDIT_DEAD_CODE_2026Q2.md
# all of those .cpp files were verified dead and deleted in PRs #159/#160:
#   - DSP primitives migrated into include/penta/common/SIMDKernels.h
#   - core/midi/harmony stubs had zero callers
#   - audio/export/project sources were never compiled into KellyCore (already
#     CMake-EXCLUDE_REGEX'd in root CMakeLists.txt)
#
# What remained: a propagation layer for the JUCE link interface and the
# common include paths. Convert to INTERFACE so consumers (KellyCore) keep
# the same PUBLIC linkage of include dirs + JUCE without trying to compile
# nonexistent sources.
#
# Dependency chain: daiw → penta → kelly  (preserved; just header-only now)

add_library(daiw_core INTERFACE)

target_include_directories(daiw_core INTERFACE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# daiw still re-exports JUCE for audio I/O / file handling / DSP types
# consumed via headers (juce::AudioBuffer, juce::AudioFormatManager, etc.).
target_link_libraries(daiw_core INTERFACE
    juce::juce_core
    juce::juce_audio_basics
    juce::juce_audio_formats
)

# INTERFACE targets don't take CXX_STANDARD properties — the consumers
# (KellyCore) already enforce C++20.
