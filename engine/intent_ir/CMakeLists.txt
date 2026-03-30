# CMakeLists.txt for Intent IR Rust crate

find_program(CARGO_EXECUTABLE cargo)
find_program(CBINDGEN_EXECUTABLE cbindgen)

option(
    INTENT_IR_REBUILD_RUST
    "Rebuild Intent IR Rust crate and regenerate FFI header with cbindgen"
    OFF
)

set(PREBUILT_RUST_LIB "${CMAKE_CURRENT_SOURCE_DIR}/target/release/libintent_ir.a")
set(FFI_HEADER "${CMAKE_BINARY_DIR}/include/intent_ir_ffi.h")
set(SOURCE_FFI_HEADER "${CMAKE_SOURCE_DIR}/src/bridge/intent_ir_ffi.h")

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/include")

if(INTENT_IR_REBUILD_RUST)
    if(NOT CARGO_EXECUTABLE)
        message(FATAL_ERROR "cargo is required when INTENT_IR_REBUILD_RUST=ON")
    endif()

    add_custom_command(
        OUTPUT ${PREBUILT_RUST_LIB}
        COMMAND ${CARGO_EXECUTABLE} build --release --locked --manifest-path ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Building Intent IR Rust crate"
        VERBATIM
    )

    if(CBINDGEN_EXECUTABLE)
        add_custom_command(
            OUTPUT ${FFI_HEADER}
            COMMAND ${CBINDGEN_EXECUTABLE} --config ${CMAKE_CURRENT_SOURCE_DIR}/cbindgen.toml --crate intent_ir --output ${FFI_HEADER}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml ${CMAKE_CURRENT_SOURCE_DIR}/cbindgen.toml
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Generating Intent IR C FFI header with cbindgen"
            VERBATIM
        )
        add_custom_target(intent_ir_rust_lib DEPENDS ${PREBUILT_RUST_LIB} ${FFI_HEADER})
    else()
        message(WARNING "cbindgen not found; using checked-in intent_ir_ffi.h")
        if(NOT EXISTS ${SOURCE_FFI_HEADER})
            message(FATAL_ERROR "Checked-in FFI header not found: ${SOURCE_FFI_HEADER}")
        endif()
        configure_file(${SOURCE_FFI_HEADER} ${FFI_HEADER} COPYONLY)
        add_custom_target(intent_ir_rust_lib DEPENDS ${PREBUILT_RUST_LIB})
    endif()
else()
    if(NOT EXISTS ${PREBUILT_RUST_LIB})
        message(FATAL_ERROR
            "Prebuilt Rust library not found at ${PREBUILT_RUST_LIB}. "
            "Build it once with `cargo build --release` in engine/intent_ir "
            "or configure with -DINTENT_IR_REBUILD_RUST=ON.")
    endif()
    if(NOT EXISTS ${SOURCE_FFI_HEADER})
        message(FATAL_ERROR "Checked-in FFI header not found: ${SOURCE_FFI_HEADER}")
    endif()
    configure_file(${SOURCE_FFI_HEADER} ${FFI_HEADER} COPYONLY)
    add_custom_target(intent_ir_rust_lib)
endif()

add_library(intent_ir_rust STATIC IMPORTED GLOBAL)
set_target_properties(intent_ir_rust PROPERTIES IMPORTED_LOCATION ${PREBUILT_RUST_LIB})
add_dependencies(intent_ir_rust intent_ir_rust_lib)

set(INTENT_IR_RUST_LIB intent_ir_rust PARENT_SCOPE)
set(INTENT_IR_FFI_HEADER ${FFI_HEADER} PARENT_SCOPE)
set(INTENT_IR_RUST_ARCHIVE ${PREBUILT_RUST_LIB} PARENT_SCOPE)
