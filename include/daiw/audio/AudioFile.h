/**
 * @file AudioFile.h
 * @brief Audio file I/O for WAV, AIFF, and FLAC formats
 *
 * Provides audio file reading and writing using libsndfile.
 * Supports:
 * - WAV (PCM, float)
 * - AIFF (optional)
 * - FLAC (optional)
 * - Sample rate conversion (stub for now)
 */

#pragma once

#include "daiw/types.hpp"
#include <string>
#include <vector>
#include <memory>

namespace daiw {
namespace audio {

/**
 * @brief Audio file format enumeration
 */
enum class AudioFormat {
    WAV,
    AIFF,
    FLAC,
    Unknown
};

/**
 * @brief Audio sample format
 */
enum class SampleFormat {
    Int16,      // 16-bit PCM
    Int24,      // 24-bit PCM
    Int32,      // 32-bit PCM
    Float32,    // 32-bit float
    Float64     // 64-bit float (double)
};

/**
 * @brief Audio file metadata
 */
struct AudioFileInfo {
    AudioFormat format = AudioFormat::Unknown;
    SampleFormat sampleFormat = SampleFormat::Float32;
    SampleRate sampleRate = DEFAULT_SAMPLE_RATE;
    uint32_t numChannels = 0;
    uint64_t numSamples = 0;     // Samples per channel
    double durationSeconds = 0.0;
    
    [[nodiscard]] uint64_t getTotalSamples() const {
        return numSamples * numChannels;
    }
};

/**
 * @brief Audio file reader/writer
 *
 * Handles reading and writing audio files in various formats.
 * Uses libsndfile for actual I/O operations.
 */
class AudioFile {
public:
    /**
     * @brief Default constructor
     */
    AudioFile();

    /**
     * @brief Destructor
     */
    ~AudioFile();

    // Prevent copying (file handle is unique)
    AudioFile(const AudioFile&) = delete;
    AudioFile& operator=(const AudioFile&) = delete;

    // Allow moving
    AudioFile(AudioFile&&) noexcept;
    AudioFile& operator=(AudioFile&&) noexcept;

    /**
     * @brief Read audio file from disk
     * @param filepath Path to audio file
     * @return true if successful
     */
    bool read(const std::string& filepath);

    /**
     * @brief Write audio file to disk
     * @param filepath Output file path
     * @param format Output format (WAV, AIFF, FLAC)
     * @param sampleFormat Sample format (Int16, Float32, etc.)
     * @return true if successful
     */
    bool write(const std::string& filepath,
               AudioFormat format = AudioFormat::WAV,
               SampleFormat sampleFormat = SampleFormat::Float32);

    /**
     * @brief Get file information
     */
    [[nodiscard]] const AudioFileInfo& getInfo() const {
        return info_;
    }

    /**
     * @brief Get audio data (interleaved)
     */
    [[nodiscard]] const std::vector<Sample>& getData() const {
        return data_;
    }

    /**
     * @brief Get audio data (mutable)
     */
    [[nodiscard]] std::vector<Sample>& getData() {
        return data_;
    }

    /**
     * @brief Set audio data from interleaved buffer
     */
    void setData(const std::vector<Sample>& data,
                 uint32_t numChannels,
                 SampleRate sampleRate);

    /**
     * @brief Get audio data for a specific channel
     * @param channel Channel index (0-based)
     * @return Vector containing channel data
     */
    [[nodiscard]] std::vector<Sample> getChannelData(uint32_t channel) const;

    /**
     * @brief Set audio data from separate channels
     * @param channels Vector of channel data (each vector is one channel)
     * @param sampleRate Sample rate
     */
    void setChannelData(const std::vector<std::vector<Sample>>& channels,
                       SampleRate sampleRate);

    /**
     * @brief Convert to a different sample rate (STUB)
     * @param targetRate Target sample rate
     * @return true if successful
     *
     * Basic sample rate conversion implemented using linear interpolation - for production, use libsamplerate
     * Options: libsamplerate, libresample, or simple linear interpolation
     */
    bool convertSampleRate(SampleRate targetRate);

    /**
     * @brief Generate a sine wave (for testing)
     * @param frequency Frequency in Hz
     * @param durationSeconds Duration in seconds
     * @param sampleRate Sample rate
     * @param amplitude Amplitude (0.0 to 1.0)
     */
    static AudioFile generateSineWave(float frequency,
                                     double durationSeconds,
                                     SampleRate sampleRate = DEFAULT_SAMPLE_RATE,
                                     float amplitude = 0.5f);

    /**
     * @brief Detect file format from extension
     */
    static AudioFormat detectFormat(const std::string& filepath);

private:
    AudioFileInfo info_;
    std::vector<Sample> data_;  // Interleaved audio data
    
    // Internal implementation detail (pImpl pattern for libsndfile handle)
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace audio
} // namespace daiw
