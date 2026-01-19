//! Intent IR v1 - Rust Implementation
//!
//! Provides validation, clamping, and safe construction of IntentFrame structures.

#![no_std]

extern crate alloc;

// Global allocator for no_std
use alloc::alloc::GlobalAlloc;
use alloc::alloc::Layout;

struct DummyAllocator;

unsafe impl GlobalAlloc for DummyAllocator {
    unsafe fn alloc(&self, _layout: Layout) -> *mut u8 {
        core::ptr::null_mut()
    }
    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {}
}

#[global_allocator]
static ALLOCATOR: DummyAllocator = DummyAllocator;

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
