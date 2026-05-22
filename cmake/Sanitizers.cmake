# Sanitizers.cmake — AddressSanitizer, UndefinedBehaviorSanitizer, ThreadSanitizer

function(enable_sanitizers target)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE
            -fsanitize=address,undefined
            -fno-omit-frame-pointer
        )
        target_link_options(${target} PRIVATE
            -fsanitize=address,undefined
        )
    endif()
endfunction()

# ThreadSanitizer is mutually exclusive with ASan.
function(enable_thread_sanitizer target)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE
            -fsanitize=thread
            -fno-omit-frame-pointer
        )
        target_link_options(${target} PRIVATE
            -fsanitize=thread
        )
    endif()
endfunction()

# RealtimeSanitizer (LLVM 20+) — detects heap allocation, locks, blocking IO,
# and other realtime-violating operations in functions marked
# [[clang::nonblocking]] or [[clang::nonallocating]].
#
# Apple Clang lags upstream; you may need Homebrew LLVM:
#     brew install llvm
#     CC=/opt/homebrew/opt/llvm/bin/clang \
#     CXX=/opt/homebrew/opt/llvm/bin/clang++ \
#     cmake -S . -B build-rtsan -DCMAKE_BUILD_TYPE=Debug \
#           -DBUILD_KELLY_FFI=ON -DKMIDI_ENABLE_RTSAN=ON
#
# Mutually exclusive with ASan and TSan (the runtimes overlap).
function(enable_realtime_sanitizer target)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(${target} PRIVATE
            -fsanitize=realtime
            -fno-omit-frame-pointer
        )
        target_link_options(${target} PRIVATE
            -fsanitize=realtime
        )
        target_compile_definitions(${target} PRIVATE
            KMIDI_RTSAN_ENABLED=1
        )
    endif()
endfunction()
