//! Intent IR v1 - Rust Implementation
//!
//! Provides validation, clamping, and safe construction of IntentFrame structures.

#![no_std]

extern crate alloc;

// Global allocator for no_std - use system allocator via libc
use alloc::alloc::{GlobalAlloc, Layout};

struct SystemAllocator;

unsafe impl GlobalAlloc for SystemAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // Use libc malloc for actual allocation
        extern "C" {
            fn malloc(size: usize) -> *mut u8;
        }
        malloc(layout.size())
    }

    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        extern "C" {
            fn free(ptr: *mut u8);
        }
        free(ptr);
    }
}

#[global_allocator]
static ALLOCATOR: SystemAllocator = SystemAllocator;

// Panic handler for no_std (abort on panic)
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

pub mod types;
pub mod validator;
pub mod builder;
pub mod ffi;

pub use types::*;
pub use validator::*;
pub use builder::*;
