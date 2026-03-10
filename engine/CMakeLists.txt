cmake_minimum_required(VERSION 3.20)
project(kmidi_engine LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Static library: C ABI for RT harness and plugin/FFI consumers.
add_library(engine STATIC
  src/engine.c
)

target_include_directories(engine
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(engine PRIVATE
  $<$<C_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra>
  $<$<C_COMPILER_ID:MSVC>:/W4>
)

install(TARGETS engine
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(FILES include/engine.h
  DESTINATION include/engine
)
