# DAiW File I/O & MIDI Foundation Build Configuration
cmake_minimum_required(VERSION 3.17)
project(daiw_fileio VERSION 1.0.0 LANGUAGES CXX)

# C++17 required for <filesystem>
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Source files for File I/O & MIDI library
set(FILEIO_SOURCES
    src/midi/MidiMessage.cpp
    src/midi/MidiSequence.cpp
    src/midi/MidiIO.cpp
    src/audio/AudioFile.cpp
    src/project/ProjectFile.cpp
    src/export/StemExporter.cpp
)

# Create library
add_library(daiw_fileio STATIC ${FILEIO_SOURCES})

target_include_directories(daiw_fileio PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Platform-specific libraries
if(UNIX AND NOT APPLE)
    target_link_libraries(daiw_fileio PUBLIC stdc++fs)  # For <filesystem> on older GCC
endif()

# Optional: Find and link external libraries if available
find_package(SndFile QUIET)
if(SndFile_FOUND)
    target_link_libraries(daiw_fileio PRIVATE SndFile::sndfile)
    target_compile_definitions(daiw_fileio PRIVATE HAVE_LIBSNDFILE)
    message(STATUS "libsndfile found - using for audio I/O")
else()
    message(STATUS "libsndfile not found - using stub WAV implementation")
endif()

find_package(RtMidi QUIET)
if(RtMidi_FOUND)
    target_link_libraries(daiw_fileio PRIVATE RtMidi::rtmidi)
    target_compile_definitions(daiw_fileio PRIVATE HAVE_RTMIDI)
    message(STATUS "RtMidi found - MIDI device I/O enabled")
else()
    message(STATUS "RtMidi not found - MIDI device I/O stubbed")
endif()

# Testing
option(BUILD_TESTS "Build unit tests" ON)

if(BUILD_TESTS)
    # Find GoogleTest
    find_package(GTest QUIET)
    
    if(GTest_FOUND)
        enable_testing()
        include(GoogleTest)
        
        # Test executables
        add_executable(MidiSequenceTest tests/MidiSequenceTest.cpp)
        target_link_libraries(MidiSequenceTest PRIVATE daiw_fileio GTest::gtest GTest::gtest_main)
        gtest_discover_tests(MidiSequenceTest)
        
        add_executable(AudioFileTest tests/AudioFileTest.cpp)
        target_link_libraries(AudioFileTest PRIVATE daiw_fileio GTest::gtest GTest::gtest_main)
        gtest_discover_tests(AudioFileTest)
        
        add_executable(StemExporterTest tests/StemExporterTest.cpp)
        target_link_libraries(StemExporterTest PRIVATE daiw_fileio GTest::gtest GTest::gtest_main)
        gtest_discover_tests(StemExporterTest)
        
        message(STATUS "Tests enabled - build with 'cmake --build . --target test'")
    else()
        message(STATUS "GoogleTest not found - tests disabled")
        message(STATUS "  Install with: sudo apt-get install libgtest-dev (Ubuntu)")
        message(STATUS "            or: brew install googletest (macOS)")
    endif()
endif()

# Installation
install(TARGETS daiw_fileio
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/daiw
    DESTINATION include
)

# Print summary
message(STATUS "")
message(STATUS "========================================")
message(STATUS "DAiW File I/O & MIDI Foundation")
message(STATUS "========================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: C++${CMAKE_CXX_STANDARD}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
message(STATUS "Components:")
message(STATUS "  MIDI Message/Sequence: YES")
message(STATUS "  Audio File I/O: ${SndFile_FOUND} (stub if NO)")
message(STATUS "  MIDI Device I/O: ${RtMidi_FOUND} (stub if NO)")
message(STATUS "  Project Files: YES")
message(STATUS "  Stem Export: YES")
message(STATUS "  Tests: ${BUILD_TESTS}")
message(STATUS "========================================")
message(STATUS "")
