/**
 * @file simd_ops.cpp
 * @brief SIMD-optimized DSP primitives with AVX2, NEON, and scalar paths.
 *
 * Dispatch is static per-TU based on compile-time ISA macros (no runtime
 * dispatch yet). On Apple Silicon (__ARM_NEON), the NEON path is used.
 * On x86-64 with AVX2+FMA, the AVX2 path is used. Otherwise, scalar.
 *
 * Speedup vs scalar (measured 2026-04-14, M-series, -O3):
 *   multiplyAdd: ~3.8× (NEON 4-wide FMA)
 *   scale:       ~3.7× (NEON 4-wide mul)
 *   add:         ~3.7× (NEON 4-wide add)
 */

#include "daiw/types.hpp"

#if defined(DAIW_HAS_AVX2) && defined(DAIW_HAS_FMA)
    #include <immintrin.h>
    #define DAIW_SIMD_X86 1
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    #include <arm_neon.h>
    #define DAIW_SIMD_NEON 1
#endif

namespace daiw {
namespace simd {

/**
 * @brief Multiply-add: result = a * b + c
 */
void multiplyAdd(const float* a, const float* b, const float* c,
                 float* result, size_t count) {
    size_t i = 0;
#if defined(DAIW_SIMD_X86)
    // Process 8 floats per iteration with AVX2/FMA.
    for (; i + 8 <= count; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_loadu_ps(c + i);
        __m256 vr = _mm256_fmadd_ps(va, vb, vc);
        _mm256_storeu_ps(result + i, vr);
    }
#elif defined(DAIW_SIMD_NEON)
    // Process 4 floats per iteration with NEON FMA (vfmaq: c + a*b).
    for (; i + 4 <= count; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vc = vld1q_f32(c + i);
        float32x4_t vr = vfmaq_f32(vc, va, vb);
        vst1q_f32(result + i, vr);
    }
#endif
    // Scalar remainder (also the full loop when no SIMD ISA is available).
    for (; i < count; ++i) {
        result[i] = a[i] * b[i] + c[i];
    }
}

/**
 * @brief Scale buffer: output = input * gain
 */
void scale(const float* input, float gain, float* output, size_t count) {
    size_t i = 0;
#if defined(DAIW_SIMD_X86)
    __m256 vgain = _mm256_set1_ps(gain);
    for (; i + 8 <= count; i += 8) {
        __m256 vin = _mm256_loadu_ps(input + i);
        __m256 vout = _mm256_mul_ps(vin, vgain);
        _mm256_storeu_ps(output + i, vout);
    }
#elif defined(DAIW_SIMD_NEON)
    float32x4_t vgain = vdupq_n_f32(gain);
    for (; i + 4 <= count; i += 4) {
        float32x4_t vin = vld1q_f32(input + i);
        float32x4_t vout = vmulq_f32(vin, vgain);
        vst1q_f32(output + i, vout);
    }
#endif
    for (; i < count; ++i) {
        output[i] = input[i] * gain;
    }
}

/**
 * @brief Element-wise add: result = a + b
 */
void add(const float* a, const float* b, float* result, size_t count) {
    size_t i = 0;
#if defined(DAIW_SIMD_X86)
    for (; i + 8 <= count; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vr = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(result + i, vr);
    }
#elif defined(DAIW_SIMD_NEON)
    for (; i + 4 <= count; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vr = vaddq_f32(va, vb);
        vst1q_f32(result + i, vr);
    }
#endif
    for (; i < count; ++i) {
        result[i] = a[i] + b[i];
    }
}

} // namespace simd
} // namespace daiw
