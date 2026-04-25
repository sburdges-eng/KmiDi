#pragma once

/**
 * Platform compatibility layer for Penta-Core
 * Handles macOS SDK compatibility issues and cross-platform definitions
 *
 * IMPORTANT: This file MUST be included before any other system headers!
 */

// ============================================================================
// macOS SDK availability macro handling
// Include system Availability.h FIRST to properly define all availability macros
// before any other system headers are included
// ============================================================================
#if defined(__APPLE__) || defined(__APPLE_CPP__) || defined(__APPLE_CC__)

#include <Availability.h>
#include <AvailabilityMacros.h>
#include <TargetConditionals.h>

#endif // __APPLE__

// Standard includes - safe after the above definitions
#include <cstddef>
#include <cstdint>

// ============================================================================
// T6.4 — Denormal suppression helper
//
// Call set_denormals_off() once per thread at thread entry (worker threads,
// audio render thread).  Prevents 100x slowdown on near-silence audio that
// produces denormal IEEE-754 values in biquad / convolution state variables.
//
// x86_64:   sets FTZ (bit 15) and DAZ (bit 6) in MXCSR.
// AArch64:  sets FPCR.FZ (bit 24) — Flush-to-Zero.
// Other:    no-op (scalar paths tolerate denormals at the cost of extra cycles).
// ============================================================================
#if defined(__SSE__)
#include <immintrin.h>
#endif

namespace penta {

inline void set_denormals_off() noexcept {
#if defined(__SSE__)
    _mm_setcsr(_mm_getcsr() | 0x8040u);  // FTZ (bit 15) + DAZ (bit 6)
#elif defined(__aarch64__)
    std::uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    fpcr |= (1ull << 24);               // FPCR.FZ = Flush-to-Zero
    __asm__ volatile("msr fpcr, %0" : : "r"(fpcr));
#endif
    // Other architectures: no-op — scalar paths are unaffected.
}

} // namespace penta
