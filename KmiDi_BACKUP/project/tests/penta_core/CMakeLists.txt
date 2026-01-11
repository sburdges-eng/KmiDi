cmake_minimum_required(VERSION 3.20)
project(PentaCoreTests)

include(GoogleTest)

# Test sources
set(TEST_SOURCES
    harmony_test.cpp
    groove_test.cpp
    osc_test.cpp
    rt_memory_test.cpp
    performance_test.cpp
    diagnostics_test.cpp
    plugin_test_harness.cpp
)

add_executable(penta_tests ${TEST_SOURCES})

# JUCE's juce_core uses libcurl on Linux for networking utilities; when JUCE
# modules are compiled into the test binary we must link curl explicitly.
find_package(CURL REQUIRED)

target_link_libraries(penta_tests PRIVATE
    penta_core
    gtest
    gtest_main
    CURL::libcurl
)

# Enable SIMD optimizations for performance tests
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
if(COMPILER_SUPPORTS_AVX2)
    target_compile_options(penta_tests PRIVATE -mavx2 -mpopcnt -mfma)
endif()

# Discover tests for CTest
gtest_discover_tests(penta_tests)
