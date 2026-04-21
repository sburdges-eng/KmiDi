// NEON/AVX2 vs scalar parity tests for penta::simd free-function primitives.
//
// Standalone executable — no Catch2, no KellyCore, no JUCE. Header-only.
// Matches the idiom of test_rt_state_seqlock.cpp: a plain main() that
// returns 0 on success and 1 on any assertion failure.
//
// For each primitive (dot_product_f32, multiply_add_f32, max_element_f32)
// this file:
//  - generates deterministic pseudorandom float buffers (LCG seeded to 42)
//  - calls the SIMD primitive and a known-good scalar reference
//  - verifies |simd - scalar| < epsilon (1e-5) for edge sizes:
//    {0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 1024}
//
// The SIMD path selected at compile time is whichever of AVX2 / NEON /
// scalar the compiler enables.  On Apple Silicon (arm64) that is NEON.

// Force the header to compile in this TU exactly as callers will see it.
#include "penta/common/SIMDKernels.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// ---------------------------------------------------------------------------
// Tiny deterministic pseudorandom float generator (LCG, period 2^32).
// Produces values in (-1, 1).
// ---------------------------------------------------------------------------
static unsigned int lcg_state = 42u;

static float lcg_next() {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    // Map [0, UINT_MAX] -> (-1, 1)
    return (static_cast<float>(lcg_state) / static_cast<float>(0xFFFFFFFFu)) * 2.0f - 1.0f;
}

static void fill_random(float* buf, size_t n) {
    for (size_t i = 0; i < n; ++i) buf[i] = lcg_next();
}

// ---------------------------------------------------------------------------
// Scalar reference implementations (correctness baseline).
// These intentionally duplicate the scalar path so the test is self-contained
// and does not rely on the header's own scalar fallback being correct.
// ---------------------------------------------------------------------------

static float ref_dot_product(const float* a, const float* b, size_t n) {
    float s = 0.0f;
    for (size_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

static void ref_multiply_add(float* dst, const float* a, const float* b, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] += a[i] * b[i];
}

static float ref_max_element(const float* a, size_t n) {
    if (n == 0) return -__builtin_inff();
    float m = a[0];
    for (size_t i = 1; i < n; ++i) if (a[i] > m) m = a[i];
    return m;
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

static constexpr float kEpsilon = 1e-5f;

static bool near(float a, float b) {
    return std::fabs(a - b) < kEpsilon;
}

static int failures = 0;

#define EXPECT_NEAR(val, ref, label, n)                                         \
    do {                                                                        \
        if (!near((val), (ref))) {                                              \
            std::printf("FAIL  %s  n=%-5zu  simd=%.8f  ref=%.8f  diff=%.2e\n", \
                (label), (size_t)(n), (double)(val), (double)(ref),             \
                (double)std::fabs((val) - (ref)));                              \
            ++failures;                                                         \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Edge sizes to cover: straddle every SIMD lane boundary.
// ---------------------------------------------------------------------------
static const size_t kSizes[] = { 0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 1024 };
static constexpr size_t kNSizes = sizeof(kSizes) / sizeof(kSizes[0]);
static constexpr size_t kMaxN   = 1024;

// ---------------------------------------------------------------------------
// dot_product_f32
// ---------------------------------------------------------------------------
static void test_dot_product() {
    static float a[kMaxN], b[kMaxN];

    for (size_t si = 0; si < kNSizes; ++si) {
        const size_t n = kSizes[si];

        lcg_state = 42u;
        fill_random(a, n);
        fill_random(b, n);

        float simd_result = penta::simd::dot_product_f32(a, b, n);

        lcg_state = 42u;
        fill_random(a, n);
        fill_random(b, n);
        float ref_result = ref_dot_product(a, b, n);

        EXPECT_NEAR(simd_result, ref_result, "dot_product_f32", n);
    }
}

// ---------------------------------------------------------------------------
// multiply_add_f32
// ---------------------------------------------------------------------------
static void test_multiply_add() {
    static float dst_simd[kMaxN], dst_ref[kMaxN];
    static float a[kMaxN], b[kMaxN];

    for (size_t si = 0; si < kNSizes; ++si) {
        const size_t n = kSizes[si];

        // Fill dst, a, b identically for both paths.
        lcg_state = 99u;
        fill_random(dst_simd, n);
        fill_random(a, n);
        fill_random(b, n);

        // Copy dst into ref path before mutating
        for (size_t i = 0; i < n; ++i) dst_ref[i] = dst_simd[i];

        // Recompute a, b for ref with same seed
        lcg_state = 99u;
        fill_random(dst_ref, n);   // overwrite — same values, just advancing LCG
        float tmp_a[kMaxN], tmp_b[kMaxN];
        fill_random(tmp_a, n);
        fill_random(tmp_b, n);

        // Accumulate both
        lcg_state = 99u;
        fill_random(dst_simd, n);
        fill_random(a, n);
        fill_random(b, n);
        penta::simd::multiply_add_f32(dst_simd, a, b, n);

        lcg_state = 99u;
        fill_random(dst_ref, n);
        fill_random(tmp_a, n);
        fill_random(tmp_b, n);
        ref_multiply_add(dst_ref, tmp_a, tmp_b, n);

        for (size_t i = 0; i < n; ++i) {
            if (!near(dst_simd[i], dst_ref[i])) {
                std::printf("FAIL  multiply_add_f32  n=%-5zu  i=%-5zu"
                            "  simd=%.8f  ref=%.8f  diff=%.2e\n",
                            n, i,
                            (double)dst_simd[i], (double)dst_ref[i],
                            (double)std::fabs(dst_simd[i] - dst_ref[i]));
                ++failures;
                break; // one failure per size is enough
            }
        }
    }
}

// ---------------------------------------------------------------------------
// max_element_f32
// ---------------------------------------------------------------------------
static void test_max_element() {
    static float a[kMaxN];

    // Edge case: n == 0 should return -inf
    {
        float simd_result = penta::simd::max_element_f32(a, 0);
        // Only check it's not a finite number bigger than our data
        if (simd_result != -__builtin_inff() && !(simd_result < -1e30f)) {
            std::printf("FAIL  max_element_f32  n=0  expected -inf, got %.8f\n",
                        (double)simd_result);
            ++failures;
        }
    }

    for (size_t si = 1; si < kNSizes; ++si) {  // skip n==0 (tested above)
        const size_t n = kSizes[si];
        if (n == 0) continue;

        lcg_state = 7u;
        fill_random(a, n);

        float simd_result = penta::simd::max_element_f32(a, n);

        lcg_state = 7u;
        fill_random(a, n);
        float ref_result = ref_max_element(a, n);

        EXPECT_NEAR(simd_result, ref_result, "max_element_f32", n);
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    test_dot_product();
    test_multiply_add();
    test_max_element();

    if (failures == 0) {
        std::printf("PASS  all penta::simd kernel parity tests (%zu sizes)\n", kNSizes);
        return 0;
    } else {
        std::printf("FAIL  %d penta::simd kernel parity test(s) failed\n", failures);
        return 1;
    }
}
