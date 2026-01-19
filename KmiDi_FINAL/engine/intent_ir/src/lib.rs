//! Intent IR v1 - Rust Implementation
//!
//! Provides validation, clamping, and safe construction of IntentFrame structures.

#![no_std]

extern crate alloc;

pub mod types;
pub mod validator;
pub mod builder;
pub mod ffi;

pub use types::*;
pub use validator::*;
pub use builder::*;
