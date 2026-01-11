#pragma once

#include "penta/common/Platform.h"

#include <array>
#include <cstddef>
#include <cmath>

// SIMD intrinsics - detect availability
#if defined(__AVX2__)
    #include <immintrin.h>
    #define PENTA_HAS_AVX2 1
#elif defined(__SSE4_1__)
    #include <smmintrin.h>
    #define PENTA_HAS_SSE4 1
#elif defined(__SSE2__)
    #include <emmintrin.h>
    #define PENTA_HAS_SSE2 1
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    #include <arm_neon.h>
    #define PENTA_HAS_NEON 1
#endif

namespace penta {

/**
 * SIMD-optimized DSP kernels with scalar fallbacks.
 * All functions are marked noexcept for RT-safety.
 */
class SIMDKernels {
public:
    // =========================================================================
    // RMS Calculation
    // =========================================================================

    /**
     * Calculate RMS of a buffer using SIMD when available.
     * Processes 8 samples at a time with AVX, 4 with SSE.
     *
     * @param buffer Input audio samples
     * @param frames Number of samples
     * @return RMS value
     */
    static float calculateRMS(const float* buffer, size_t frames) noexcept {
#if defined(PENTA_HAS_AVX2)
        return calculateRMS_AVX2(buffer, frames);
#elif defined(PENTA_HAS_SSE2)
        return calculateRMS_SSE2(buffer, frames);
#elif defined(PENTA_HAS_NEON)
        return calculateRMS_NEON(buffer, frames);
#else
        return calculateRMS_Scalar(buffer, frames);
#endif
    }

    // =========================================================================
    // Sum of Squares (for energy calculation)
    // =========================================================================

    /**
     * Calculate sum of squares of a buffer.
     *
     * @param buffer Input audio samples
     * @param frames Number of samples
     * @return Sum of squares
     */
    static float sumOfSquares(const float* buffer, size_t frames) noexcept {
#if defined(PENTA_HAS_AVX2)
        return sumOfSquares_AVX2(buffer, frames);
#elif defined(PENTA_HAS_SSE2)
        return sumOfSquares_SSE2(buffer, frames);
#elif defined(PENTA_HAS_NEON)
        return sumOfSquares_NEON(buffer, frames);
#else
        return sumOfSquares_Scalar(buffer, frames);
#endif
    }

    // =========================================================================
    // Spectral Flux (positive difference sum)
    // =========================================================================

    /**
     * Calculate spectral flux between two spectra.
     * Returns sum of positive differences.
     *
     * @param current Current spectrum
     * @param previous Previous spectrum
     * @param size Number of bins
     * @return Spectral flux value
     */
    static float spectralFlux(const float* current, const float* previous, size_t size) noexcept {
#if defined(PENTA_HAS_AVX2)
        return spectralFlux_AVX2(current, previous, size);
#elif defined(PENTA_HAS_SSE2)
        return spectralFlux_SSE2(current, previous, size);
#elif defined(PENTA_HAS_NEON)
        return spectralFlux_NEON(current, previous, size);
#else
        return spectralFlux_Scalar(current, previous, size);
#endif
    }

    // =========================================================================
    // Apply Window Function
    // =========================================================================

    /**
     * Apply window function to buffer (multiply in-place).
     *
     * @param buffer Audio buffer (modified in-place)
     * @param window Window coefficients
     * @param frames Number of samples
     */
    static void applyWindow(float* buffer, const float* window, size_t frames) noexcept {
#if defined(PENTA_HAS_AVX2)
        applyWindow_AVX2(buffer, window, frames);
#elif defined(PENTA_HAS_SSE2)
        applyWindow_SSE2(buffer, window, frames);
#elif defined(PENTA_HAS_NEON)
        applyWindow_NEON(buffer, window, frames);
#else
        applyWindow_Scalar(buffer, window, frames);
#endif
    }

    // =========================================================================
    // Autocorrelation
    // =========================================================================

    /**
     * Calculate autocorrelation at a specific lag.
     *
     * @param buffer Input signal
     * @param size Signal length
     * @param lag Lag in samples
     * @return Autocorrelation value at lag
     */
    static float autocorrelationAtLag(const float* buffer, size_t size, size_t lag) noexcept {
#if defined(PENTA_HAS_AVX2)
        return autocorrelationAtLag_AVX2(buffer, size, lag);
#elif defined(PENTA_HAS_SSE2)
        return autocorrelationAtLag_SSE2(buffer, size, lag);
#elif defined(PENTA_HAS_NEON)
        return autocorrelationAtLag_NEON(buffer, size, lag);
#else
        return autocorrelationAtLag_Scalar(buffer, size, lag);
#endif
    }

    // =========================================================================
    // Dot Product
    // =========================================================================

    /**
     * Calculate dot product of two vectors.
     *
     * @param a First vector
     * @param b Second vector
     * @param size Vector length
     * @return Dot product
     */
    static float dotProduct(const float* a, const float* b, size_t size) noexcept {
#if defined(PENTA_HAS_AVX2)
        return dotProduct_AVX2(a, b, size);
#elif defined(PENTA_HAS_SSE2)
        return dotProduct_SSE2(a, b, size);
#elif defined(PENTA_HAS_NEON)
        return dotProduct_NEON(a, b, size);
#else
        return dotProduct_Scalar(a, b, size);
#endif
    }

private:
    // =========================================================================
    // Scalar Implementations (Fallback)
    // =========================================================================

    static float calculateRMS_Scalar(const float* buffer, size_t frames) noexcept {
        if (frames == 0) return 0.0f;
        float sum = sumOfSquares_Scalar(buffer, frames);
        return std::sqrt(sum / static_cast<float>(frames));
    }

    static float sumOfSquares_Scalar(const float* buffer, size_t frames) noexcept {
        float sum = 0.0f;
        for (size_t i = 0; i < frames; ++i) {
            sum += buffer[i] * buffer[i];
        }
        return sum;
    }

    static float spectralFlux_Scalar(const float* current, const float* previous, size_t size) noexcept {
        float flux = 0.0f;
        for (size_t i = 0; i < size; ++i) {
            float diff = current[i] - previous[i];
            if (diff > 0.0f) {
                flux += diff;
            }
        }
        return flux;
    }

    static void applyWindow_Scalar(float* buffer, const float* window, size_t frames) noexcept {
        for (size_t i = 0; i < frames; ++i) {
            buffer[i] *= window[i];
        }
    }

    static float autocorrelationAtLag_Scalar(const float* buffer, size_t size, size_t lag) noexcept {
        if (lag >= size) return 0.0f;
        float sum = 0.0f;
        size_t end = size - lag;
        for (size_t i = 0; i < end; ++i) {
            sum += buffer[i] * buffer[i + lag];
        }
        return sum / static_cast<float>(end);
    }

    static float dotProduct_Scalar(const float* a, const float* b, size_t size) noexcept {
        float sum = 0.0f;
        for (size_t i = 0; i < size; ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    }

#ifdef PENTA_HAS_AVX2
    // =========================================================================
    // AVX2 Implementations
    // =========================================================================

    static float calculateRMS_AVX2(const float* buffer, size_t frames) noexcept {
        if (frames == 0) return 0.0f;
        float sum = sumOfSquares_AVX2(buffer, frames);
        return std::sqrt(sum / static_cast<float>(frames));
    }

    static float sumOfSquares_AVX2(const float* buffer, size_t frames) noexcept {
        __m256 vSum = _mm256_setzero_ps();

        // Process 8 floats at a time
        size_t i = 0;
        size_t simdEnd = frames - (frames % 8);

        for (; i < simdEnd; i += 8) {
            __m256 v = _mm256_loadu_ps(buffer + i);
            vSum = _mm256_fmadd_ps(v, v, vSum);  // vSum += v * v
        }

        // Horizontal sum of AVX register
        __m128 vLow = _mm256_castps256_ps128(vSum);
        __m128 vHigh = _mm256_extractf128_ps(vSum, 1);
        vLow = _mm_add_ps(vLow, vHigh);

        __m128 shuf = _mm_shuffle_ps(vLow, vLow, _MM_SHUFFLE(2, 3, 0, 1));
        __m128 sums = _mm_add_ps(vLow, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        float sum = _mm_cvtss_f32(sums);

        // Handle remaining elements
        for (; i < frames; ++i) {
            sum += buffer[i] * buffer[i];
        }

        return sum;
    }

    static float spectralFlux_AVX2(const float* current, const float* previous, size_t size) noexcept {
        __m256 vFlux = _mm256_setzero_ps();
        __m256 vZero = _mm256_setzero_ps();

        size_t i = 0;
        size_t simdEnd = size - (size % 8);

        for (; i < simdEnd; i += 8) {
            __m256 vCurrent = _mm256_loadu_ps(current + i);
            __m256 vPrevious = _mm256_loadu_ps(previous + i);
            __m256 vDiff = _mm256_sub_ps(vCurrent, vPrevious);

            // Keep only positive differences: max(diff, 0)
            __m256 vPositive = _mm256_max_ps(vDiff, vZero);
            vFlux = _mm256_add_ps(vFlux, vPositive);
        }

        // Horizontal sum
        __m128 vLow = _mm256_castps256_ps128(vFlux);
        __m128 vHigh = _mm256_extractf128_ps(vFlux, 1);
        vLow = _mm_add_ps(vLow, vHigh);

        __m128 shuf = _mm_shuffle_ps(vLow, vLow, _MM_SHUFFLE(2, 3, 0, 1));
        __m128 sums = _mm_add_ps(vLow, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        float flux = _mm_cvtss_f32(sums);

        // Handle remaining elements
        for (; i < size; ++i) {
            float diff = current[i] - previous[i];
            if (diff > 0.0f) {
                flux += diff;
            }
        }

        return flux;
    }

    static void applyWindow_AVX2(float* buffer, const float* window, size_t frames) noexcept {
        size_t i = 0;
        size_t simdEnd = frames - (frames % 8);

        for (; i < simdEnd; i += 8) {
            __m256 vBuffer = _mm256_loadu_ps(buffer + i);
            __m256 vWindow = _mm256_loadu_ps(window + i);
            __m256 vResult = _mm256_mul_ps(vBuffer, vWindow);
            _mm256_storeu_ps(buffer + i, vResult);
        }

        // Handle remaining elements
        for (; i < frames; ++i) {
            buffer[i] *= window[i];
        }
    }

    static float autocorrelationAtLag_AVX2(const float* buffer, size_t size, size_t lag) noexcept {
        if (lag >= size) return 0.0f;

        size_t end = size - lag;
        return dotProduct_AVX2(buffer, buffer + lag, end) / static_cast<float>(end);
    }

    static float dotProduct_AVX2(const float* a, const float* b, size_t size) noexcept {
        __m256 vSum = _mm256_setzero_ps();

        size_t i = 0;
        size_t simdEnd = size - (size % 8);

        for (; i < simdEnd; i += 8) {
            __m256 vA = _mm256_loadu_ps(a + i);
            __m256 vB = _mm256_loadu_ps(b + i);
            vSum = _mm256_fmadd_ps(vA, vB, vSum);
        }

        // Horizontal sum
        __m128 vLow = _mm256_castps256_ps128(vSum);
        __m128 vHigh = _mm256_extractf128_ps(vSum, 1);
        vLow = _mm_add_ps(vLow, vHigh);

        __m128 shuf = _mm_shuffle_ps(vLow, vLow, _MM_SHUFFLE(2, 3, 0, 1));
        __m128 sums = _mm_add_ps(vLow, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        float sum = _mm_cvtss_f32(sums);

        // Handle remaining elements
        for (; i < size; ++i) {
            sum += a[i] * b[i];
        }

        return sum;
    }
#endif // PENTA_HAS_AVX2

#ifdef PENTA_HAS_SSE2
    // =========================================================================
    // SSE2 Implementations
    // =========================================================================

    static float calculateRMS_SSE2(const float* buffer, size_t frames) noexcept {
        if (frames == 0) return 0.0f;
        float sum = sumOfSquares_SSE2(buffer, frames);
        return std::sqrt(sum / static_cast<float>(frames));
    }

    static float sumOfSquares_SSE2(const float* buffer, size_t frames) noexcept {
        __m128 vSum = _mm_setzero_ps();

        size_t i = 0;
        size_t simdEnd = frames - (frames % 4);

        for (; i < simdEnd; i += 4) {
            __m128 v = _mm_loadu_ps(buffer + i);
            __m128 vSq = _mm_mul_ps(v, v);
            vSum = _mm_add_ps(vSum, vSq);
        }

        // Horizontal sum
        __m128 shuf = _mm_shuffle_ps(vSum, vSum, _MM_SHUFFLE(2, 3, 0, 1));
        __m128 sums = _mm_add_ps(vSum, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        float sum = _mm_cvtss_f32(sums);

        // Handle remaining elements
        for (; i < frames; ++i) {
            sum += buffer[i] * buffer[i];
        }

        return sum;
    }

    static float spectralFlux_SSE2(const float* current, const float* previous, size_t size) noexcept {
        __m128 vFlux = _mm_setzero_ps();
        __m128 vZero = _mm_setzero_ps();

        size_t i = 0;
        size_t simdEnd = size - (size % 4);

        for (; i < simdEnd; i += 4) {
            __m128 vCurrent = _mm_loadu_ps(current + i);
            __m128 vPrevious = _mm_loadu_ps(previous + i);
            __m128 vDiff = _mm_sub_ps(vCurrent, vPrevious);
            __m128 vPositive = _mm_max_ps(vDiff, vZero);
            vFlux = _mm_add_ps(vFlux, vPositive);
        }

        // Horizontal sum
        __m128 shuf = _mm_shuffle_ps(vFlux, vFlux, _MM_SHUFFLE(2, 3, 0, 1));
        __m128 sums = _mm_add_ps(vFlux, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        float flux = _mm_cvtss_f32(sums);

        // Handle remaining elements
        for (; i < size; ++i) {
            float diff = current[i] - previous[i];
            if (diff > 0.0f) {
                flux += diff;
            }
        }

        return flux;
    }

    static void applyWindow_SSE2(float* buffer, const float* window, size_t frames) noexcept {
        size_t i = 0;
        size_t simdEnd = frames - (frames % 4);

        for (; i < simdEnd; i += 4) {
            __m128 vBuffer = _mm_loadu_ps(buffer + i);
            __m128 vWindow = _mm_loadu_ps(window + i);
            __m128 vResult = _mm_mul_ps(vBuffer, vWindow);
            _mm_storeu_ps(buffer + i, vResult);
        }

        for (; i < frames; ++i) {
            buffer[i] *= window[i];
        }
    }

    static float autocorrelationAtLag_SSE2(const float* buffer, size_t size, size_t lag) noexcept {
        if (lag >= size) return 0.0f;
        size_t end = size - lag;
        return dotProduct_SSE2(buffer, buffer + lag, end) / static_cast<float>(end);
    }

    static float dotProduct_SSE2(const float* a, const float* b, size_t size) noexcept {
        __m128 vSum = _mm_setzero_ps();

        size_t i = 0;
        size_t simdEnd = size - (size % 4);

        for (; i < simdEnd; i += 4) {
            __m128 vA = _mm_loadu_ps(a + i);
            __m128 vB = _mm_loadu_ps(b + i);
            __m128 vProd = _mm_mul_ps(vA, vB);
            vSum = _mm_add_ps(vSum, vProd);
        }

        // Horizontal sum
        __m128 shuf = _mm_shuffle_ps(vSum, vSum, _MM_SHUFFLE(2, 3, 0, 1));
        __m128 sums = _mm_add_ps(vSum, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        float sum = _mm_cvtss_f32(sums);

        for (; i < size; ++i) {
            sum += a[i] * b[i];
        }

        return sum;
    }
#endif // PENTA_HAS_SSE2

#ifdef PENTA_HAS_NEON
    // =========================================================================
    // ARM NEON Implementations (for Apple Silicon / ARM processors)
    // =========================================================================

    static float calculateRMS_NEON(const float* buffer, size_t frames) noexcept {
        if (frames == 0) return 0.0f;
        float sum = sumOfSquares_NEON(buffer, frames);
        return std::sqrt(sum / static_cast<float>(frames));
    }

    static float sumOfSquares_NEON(const float* buffer, size_t frames) noexcept {
        float32x4_t vSum = vdupq_n_f32(0.0f);

        size_t i = 0;
        size_t simdEnd = frames - (frames % 4);

        for (; i < simdEnd; i += 4) {
            float32x4_t v = vld1q_f32(buffer + i);
            vSum = vmlaq_f32(vSum, v, v);  // vSum += v * v
        }

        // Horizontal sum
        float32x2_t vLow = vget_low_f32(vSum);
        float32x2_t vHigh = vget_high_f32(vSum);
        float32x2_t vPair = vadd_f32(vLow, vHigh);
        float sum = vget_lane_f32(vpadd_f32(vPair, vPair), 0);

        for (; i < frames; ++i) {
            sum += buffer[i] * buffer[i];
        }

        return sum;
    }

    static float spectralFlux_NEON(const float* current, const float* previous, size_t size) noexcept {
        float32x4_t vFlux = vdupq_n_f32(0.0f);
        float32x4_t vZero = vdupq_n_f32(0.0f);

        size_t i = 0;
        size_t simdEnd = size - (size % 4);

        for (; i < simdEnd; i += 4) {
            float32x4_t vCurrent = vld1q_f32(current + i);
            float32x4_t vPrevious = vld1q_f32(previous + i);
            float32x4_t vDiff = vsubq_f32(vCurrent, vPrevious);
            float32x4_t vPositive = vmaxq_f32(vDiff, vZero);
            vFlux = vaddq_f32(vFlux, vPositive);
        }

        // Horizontal sum
        float32x2_t vLow = vget_low_f32(vFlux);
        float32x2_t vHigh = vget_high_f32(vFlux);
        float32x2_t vPair = vadd_f32(vLow, vHigh);
        float flux = vget_lane_f32(vpadd_f32(vPair, vPair), 0);

        for (; i < size; ++i) {
            float diff = current[i] - previous[i];
            if (diff > 0.0f) {
                flux += diff;
            }
        }

        return flux;
    }

    static void applyWindow_NEON(float* buffer, const float* window, size_t frames) noexcept {
        size_t i = 0;
        size_t simdEnd = frames - (frames % 4);

        for (; i < simdEnd; i += 4) {
            float32x4_t vBuffer = vld1q_f32(buffer + i);
            float32x4_t vWindow = vld1q_f32(window + i);
            float32x4_t vResult = vmulq_f32(vBuffer, vWindow);
            vst1q_f32(buffer + i, vResult);
        }

        for (; i < frames; ++i) {
            buffer[i] *= window[i];
        }
    }

    static float autocorrelationAtLag_NEON(const float* buffer, size_t size, size_t lag) noexcept {
        if (lag >= size) return 0.0f;
        size_t end = size - lag;
        return dotProduct_NEON(buffer, buffer + lag, end) / static_cast<float>(end);
    }

    static float dotProduct_NEON(const float* a, const float* b, size_t size) noexcept {
        float32x4_t vSum = vdupq_n_f32(0.0f);

        size_t i = 0;
        size_t simdEnd = size - (size % 4);

        for (; i < simdEnd; i += 4) {
            float32x4_t vA = vld1q_f32(a + i);
            float32x4_t vB = vld1q_f32(b + i);
            vSum = vmlaq_f32(vSum, vA, vB);
        }

        // Horizontal sum
        float32x2_t vLow = vget_low_f32(vSum);
        float32x2_t vHigh = vget_high_f32(vSum);
        float32x2_t vPair = vadd_f32(vLow, vHigh);
        float sum = vget_lane_f32(vpadd_f32(vPair, vPair), 0);

        for (; i < size; ++i) {
            sum += a[i] * b[i];
        }

        return sum;
    }
#endif // PENTA_HAS_NEON
};

} // namespace penta
