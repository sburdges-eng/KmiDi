# CMakeLists.txt for Intent IR Rust crate

# Find Rust toolchain
find_program(CARGO_EXECUTABLE cargo REQUIRED)

# Set Rust target directory
set(RUST_TARGET_DIR "${CMAKE_BINARY_DIR}/rust_target")
set(RUST_LIB_DIR "${RUST_TARGET_DIR}/${CMAKE_SYSTEM_PROCESSOR}/release")

# Build Rust crate
add_custom_command(
    OUTPUT ${RUST_LIB_DIR}/libintent_ir.a
    COMMAND ${CARGO_EXECUTABLE} build --release --manifest-path ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building Intent IR Rust crate"
    VERBATIM
)

# Generate C header from Rust FFI using cbindgen
set(FFI_HEADER "${CMAKE_BINARY_DIR}/include/intent_ir_ffi.h")
add_custom_command(
    OUTPUT ${FFI_HEADER}
    COMMAND ${CARGO_EXECUTABLE} install cbindgen --root ${CMAKE_BINARY_DIR}/cbindgen_root
    COMMAND ${CMAKE_BINARY_DIR}/cbindgen_root/bin/cbindgen --config ${CMAKE_CURRENT_SOURCE_DIR}/cbindgen.toml --crate intent_ir --output ${FFI_HEADER}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml ${CMAKE_CURRENT_SOURCE_DIR}/cbindgen.toml
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating C FFI header from Rust"
    VERBATIM
)

# Create custom target
add_custom_target(intent_ir_rust_lib
    DEPENDS ${RUST_LIB_DIR}/libintent_ir.a ${FFI_HEADER}
)

# Create imported library
add_library(intent_ir_rust STATIC IMPORTED)
set_target_properties(intent_ir_rust PROPERTIES
    IMPORTED_LOCATION ${RUST_LIB_DIR}/libintent_ir.a
)
add_dependencies(intent_ir_rust intent_ir_rust_lib)

# Export variables for use in parent CMakeLists
set(INTENT_IR_RUST_LIB intent_ir_rust PARENT_SCOPE)
set(INTENT_IR_FFI_HEADER ${FFI_HEADER} PARENT_SCOPE)
