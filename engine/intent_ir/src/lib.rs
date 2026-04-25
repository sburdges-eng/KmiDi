//! Intent IR v1 - Rust Implementation
//!
//! Provides validation, clamping, and safe construction of IntentFrame structures.
//!
//! This crate was previously `#![no_std]` with a custom libc allocator and
//! panic-abort handler. That posture was removed because:
//!   - The shipped artifact is a `staticlib` linked into KellyFFI, which
//!     already depends on libc++ / libSystem — std adds no meaningful weight.
//!   - `no_std` + `panic = "unwind"` cannot compile on stable without
//!     `-Zbuild-std`, which breaks `cargo test`.
//! The FFI safety guarantee (no panic unwinding through the C boundary) is
//! preserved by `panic = "abort"` in `[profile.release]` and by wrapping
//! every `extern "C"` entry in `std::panic::catch_unwind` (see T2).

pub mod types;
pub mod validator;
pub mod builder;
pub mod ffi;

pub use types::*;
pub use validator::*;
pub use builder::*;

/// ABI drift canary — catches C header / Rust struct layout divergence at
/// compile time.  The size constant is load-bearing: bump it deliberately
/// (with a matching update to the C mirror in intent_ir_ffi.h) if the
/// struct ever grows.
const _: () = {
    const INTENT_FRAME_SIZE: usize = core::mem::size_of::<types::IntentFrame>();
    const INTENT_FRAME_ALIGN: usize = core::mem::align_of::<types::IntentFrame>();
    // Size verified by running `cargo test -- print_intent_frame_size --nocapture`
    // on 2026-04-21. Bump deliberately (with a matching update to the C mirror
    // struct in src/bridge/intent_ir_ffi.h) if the struct ever grows.
    assert!(INTENT_FRAME_SIZE == 128, "IntentFrame ABI size changed — update C mirror and bump this constant");
    assert!(INTENT_FRAME_ALIGN == 8, "IntentFrame must be 8-byte aligned (matches u64 in IntentMeta)");
};
