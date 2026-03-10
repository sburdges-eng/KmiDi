cmake_minimum_required(VERSION 3.20)
project(rt_harness LANGUAGES CXX)

# Headless RT callback harness: links to engine only (no JUCE, no Qt).
add_executable(rt_harness main.cpp)

target_include_directories(rt_harness PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  $<TARGET_PROPERTY:engine,INTERFACE_INCLUDE_DIRECTORIES>
)

target_link_libraries(rt_harness PRIVATE engine)

target_compile_options(rt_harness PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra>
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

# Install so CI can run from install dir if desired
install(TARGETS rt_harness RUNTIME DESTINATION bin)
