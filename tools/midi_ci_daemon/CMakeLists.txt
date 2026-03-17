# MIDI-CI Microformat-to-SysEx daemon.
# Reads line-delimited JSON from stdin, builds MIDI 2.0 / MIDI-CI Property Exchange
# SysEx, sends via libremidi. Optional build (BUILD_MIDI_CI_DAEMON=ON).
cmake_minimum_required(VERSION 3.20)
project(midi_ci_daemon LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# libremidi: cross-platform C++20 MIDI I/O (no JUCE)
include(FetchContent)
FetchContent_Declare(
  libremidi
  GIT_REPOSITORY https://github.com/celtera/libremidi.git
  GIT_TAG        v5.0.0
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(libremidi)

add_executable(midi_ci_daemon main.cpp)

target_link_libraries(midi_ci_daemon PRIVATE libremidi)

target_compile_options(midi_ci_daemon PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra>
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

install(TARGETS midi_ci_daemon RUNTIME DESTINATION bin)
