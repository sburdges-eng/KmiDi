#pragma once

/**
 * MelSpectrogram — Compute fixed-shape log-mel spectrograms for JEPA inference.
 *
 * Output shape: (n_mels=128, n_frames=512) matching the ONNX model input.
 * Uses 22050 Hz internal sample rate, hop_length=512, n_fft=2048.
 *
 * All buffers are pre-allocated at construction time. No heap allocations
 * during compute().
 */

#include <cstddef>
#include <vector>

namespace penta::ml {

class MelSpectrogram {
public:
    static constexpr size_t kNMels     = 128;
    static constexpr size_t kNFrames   = 512;
    static constexpr size_t kNFft      = 2048;
    static constexpr size_t kHopLength = 512;
    static constexpr size_t kSampleRate = 22050;

    // Total input samples needed: (kNFrames - 1) * kHopLength + kNFft = 263680
    static constexpr size_t kRequiredSamples = (kNFrames - 1) * kHopLength + kNFft;

    MelSpectrogram();

    // Non-copyable, non-movable — compute() uses internal scratch buffers (not thread-safe)
    MelSpectrogram(const MelSpectrogram&) = delete;
    MelSpectrogram& operator=(const MelSpectrogram&) = delete;
    MelSpectrogram(MelSpectrogram&&) = delete;
    MelSpectrogram& operator=(MelSpectrogram&&) = delete;

    /**
     * Compute log-mel spectrogram from raw audio samples.
     *
     * @param samples  Input audio at kSampleRate (22050 Hz). Must have >= kRequiredSamples.
     * @param count    Number of input samples (must be >= kRequiredSamples).
     * @param output   Output buffer, size kNMels * kNFrames = 65536 floats.
     *                 Layout: row-major [mel_bin][frame], matching ONNX (1, 1, 128, 512).
     * @return true if successful
     */
    bool compute(const float* samples, size_t count, float* output);

private:
    void buildMelFilterbank();
    void applyWindow(const float* frame, float* windowed);
    void computeFFT(const float* windowed, float* magnitudes);

    // Pre-allocated buffers
    std::vector<float> window_;         // Hann window, size kNFft
    std::vector<float> windowed_;       // Windowed frame, size kNFft
    std::vector<float> fftReal_;        // FFT real part
    std::vector<float> fftImag_;        // FFT imaginary part
    std::vector<float> magnitudes_;     // Power spectrum, size kNFft/2+1
    std::vector<float> melFilterbank_;  // (kNMels, kNFft/2+1) row-major
    std::vector<float> melEnergies_;    // Single frame mel energies, size kNMels
};

} // namespace penta::ml
