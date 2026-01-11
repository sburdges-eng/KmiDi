/**
 * @file AudioFileTest.cpp
 * @brief Unit tests for AudioFile class
 */

#include <gtest/gtest.h>
#include "daiw/audio/AudioFile.h"
#include <cmath>
#include <filesystem>

using namespace daiw;
using namespace daiw::audio;
namespace fs = std::filesystem;

class AudioFileTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create temp directory for test files
        testDir_ = fs::temp_directory_path() / "audiofile_test";
        fs::create_directories(testDir_);
    }

    void TearDown() override {
        // Clean up test files
        if (fs::exists(testDir_)) {
            fs::remove_all(testDir_);
        }
    }

    fs::path testDir_;
};

// ============================================================================
// Basic Creation Tests
// ============================================================================

TEST_F(AudioFileTest, CreateEmpty) {
    AudioFile file;
    
    EXPECT_TRUE(file.getData().empty());
    EXPECT_EQ(file.getInfo().numChannels, 0);
    EXPECT_EQ(file.getInfo().numSamples, 0);
}

TEST_F(AudioFileTest, SetDataMono) {
    AudioFile file;
    
    std::vector<Sample> data = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f};
    file.setData(data, 1, 48000);
    
    EXPECT_EQ(file.getData().size(), 5);
    EXPECT_EQ(file.getInfo().numChannels, 1);
    EXPECT_EQ(file.getInfo().numSamples, 5);
    EXPECT_EQ(file.getInfo().sampleRate, 48000);
}

TEST_F(AudioFileTest, SetDataStereo) {
    AudioFile file;
    
    // Interleaved stereo: L R L R L R
    std::vector<Sample> data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    file.setData(data, 2, 44100);
    
    EXPECT_EQ(file.getData().size(), 6);
    EXPECT_EQ(file.getInfo().numChannels, 2);
    EXPECT_EQ(file.getInfo().numSamples, 3);
    EXPECT_EQ(file.getInfo().sampleRate, 44100);
}

// ============================================================================
// Sine Wave Generation Tests
// ============================================================================

TEST_F(AudioFileTest, GenerateSineWave) {
    auto file = AudioFile::generateSineWave(440.0f, 1.0, 48000, 0.5f);
    
    EXPECT_EQ(file.getInfo().numChannels, 1);
    EXPECT_EQ(file.getInfo().numSamples, 48000);
    EXPECT_EQ(file.getInfo().sampleRate, 48000);
    EXPECT_NEAR(file.getInfo().durationSeconds, 1.0, 0.001);
}

TEST_F(AudioFileTest, SineWaveAmplitude) {
    auto file = AudioFile::generateSineWave(1000.0f, 0.1, 48000, 0.8f);
    
    const auto& data = file.getData();
    
    // Find peak
    float peak = 0.0f;
    for (const auto& sample : data) {
        peak = std::max(peak, std::abs(sample));
    }
    
    // Peak should be close to amplitude (within numerical precision)
    EXPECT_NEAR(peak, 0.8f, 0.01f);
}

TEST_F(AudioFileTest, SineWaveFrequency) {
    // Generate 1Hz sine wave at 1000Hz sample rate
    auto file = AudioFile::generateSineWave(1.0f, 2.0, 1000, 1.0f);
    
    const auto& data = file.getData();
    
    // Should have 2 complete cycles
    // First sample should be ~0
    EXPECT_NEAR(data[0], 0.0f, 0.01f);
    
    // At 1/4 cycle (250 samples), should be ~1.0
    EXPECT_NEAR(data[250], 1.0f, 0.01f);
    
    // At 1/2 cycle (500 samples), should be ~0
    EXPECT_NEAR(data[500], 0.0f, 0.01f);
}

// ============================================================================
// Channel Data Tests
// ============================================================================

TEST_F(AudioFileTest, GetChannelData) {
    AudioFile file;
    
    // Create stereo data: L R L R L R
    std::vector<Sample> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    file.setData(data, 2, 48000);
    
    auto leftChannel = file.getChannelData(0);
    auto rightChannel = file.getChannelData(1);
    
    ASSERT_EQ(leftChannel.size(), 3);
    ASSERT_EQ(rightChannel.size(), 3);
    
    EXPECT_FLOAT_EQ(leftChannel[0], 1.0f);
    EXPECT_FLOAT_EQ(leftChannel[1], 3.0f);
    EXPECT_FLOAT_EQ(leftChannel[2], 5.0f);
    
    EXPECT_FLOAT_EQ(rightChannel[0], 2.0f);
    EXPECT_FLOAT_EQ(rightChannel[1], 4.0f);
    EXPECT_FLOAT_EQ(rightChannel[2], 6.0f);
}

TEST_F(AudioFileTest, SetChannelData) {
    AudioFile file;
    
    std::vector<std::vector<Sample>> channels(2);
    channels[0] = {1.0f, 2.0f, 3.0f};  // Left
    channels[1] = {4.0f, 5.0f, 6.0f};  // Right
    
    file.setChannelData(channels, 48000);
    
    EXPECT_EQ(file.getInfo().numChannels, 2);
    EXPECT_EQ(file.getInfo().numSamples, 3);
    
    const auto& data = file.getData();
    ASSERT_EQ(data.size(), 6);
    
    // Check interleaving
    EXPECT_FLOAT_EQ(data[0], 1.0f);  // L
    EXPECT_FLOAT_EQ(data[1], 4.0f);  // R
    EXPECT_FLOAT_EQ(data[2], 2.0f);  // L
    EXPECT_FLOAT_EQ(data[3], 5.0f);  // R
    EXPECT_FLOAT_EQ(data[4], 3.0f);  // L
    EXPECT_FLOAT_EQ(data[5], 6.0f);  // R
}

// ============================================================================
// File I/O Tests (Basic WAV)
// ============================================================================

TEST_F(AudioFileTest, WriteAndReadWAV) {
    auto testFile = testDir_ / "test.wav";
    
    // Create test audio
    auto original = AudioFile::generateSineWave(440.0f, 0.1, 48000, 0.5f);
    
    // Write to disk
    bool writeSuccess = original.write(testFile.string(), 
                                      AudioFormat::WAV,
                                      SampleFormat::Float32);
    EXPECT_TRUE(writeSuccess);
    EXPECT_TRUE(fs::exists(testFile));
    
    // Read back
    AudioFile loaded;
    bool readSuccess = loaded.read(testFile.string());
    EXPECT_TRUE(readSuccess);
    
    // Verify metadata
    EXPECT_EQ(loaded.getInfo().sampleRate, 48000);
    EXPECT_EQ(loaded.getInfo().numChannels, 1);
    EXPECT_EQ(loaded.getInfo().numSamples, original.getInfo().numSamples);
    
    // Verify data (should be very close)
    const auto& originalData = original.getData();
    const auto& loadedData = loaded.getData();
    
    ASSERT_EQ(originalData.size(), loadedData.size());
    
    for (size_t i = 0; i < originalData.size(); ++i) {
        EXPECT_NEAR(originalData[i], loadedData[i], 0.0001f);
    }
}

TEST_F(AudioFileTest, WriteStereoWAV) {
    auto testFile = testDir_ / "stereo.wav";
    
    AudioFile file;
    std::vector<std::vector<Sample>> channels(2);
    channels[0] = {0.1f, 0.2f, 0.3f};
    channels[1] = {0.4f, 0.5f, 0.6f};
    
    file.setChannelData(channels, 44100);
    
    bool success = file.write(testFile.string(),
                             AudioFormat::WAV,
                             SampleFormat::Float32);
    EXPECT_TRUE(success);
    
    // Read back
    AudioFile loaded;
    loaded.read(testFile.string());
    
    EXPECT_EQ(loaded.getInfo().numChannels, 2);
    EXPECT_EQ(loaded.getInfo().numSamples, 3);
}

// ============================================================================
// Format Detection Tests
// ============================================================================

TEST_F(AudioFileTest, DetectWAVFormat) {
    EXPECT_EQ(AudioFile::detectFormat("test.wav"), AudioFormat::WAV);
    EXPECT_EQ(AudioFile::detectFormat("test.WAV"), AudioFormat::WAV);
    EXPECT_EQ(AudioFile::detectFormat("/path/to/file.wav"), AudioFormat::WAV);
}

TEST_F(AudioFileTest, DetectAIFFFormat) {
    EXPECT_EQ(AudioFile::detectFormat("test.aiff"), AudioFormat::AIFF);
    EXPECT_EQ(AudioFile::detectFormat("test.aif"), AudioFormat::AIFF);
}

TEST_F(AudioFileTest, DetectFLACFormat) {
    EXPECT_EQ(AudioFile::detectFormat("test.flac"), AudioFormat::FLAC);
    EXPECT_EQ(AudioFile::detectFormat("test.FLAC"), AudioFormat::FLAC);
}

TEST_F(AudioFileTest, DetectUnknownFormat) {
    EXPECT_EQ(AudioFile::detectFormat("test.mp3"), AudioFormat::Unknown);
    EXPECT_EQ(AudioFile::detectFormat("test.txt"), AudioFormat::Unknown);
    EXPECT_EQ(AudioFile::detectFormat("test"), AudioFormat::Unknown);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(AudioFileTest, EmptyChannelData) {
    AudioFile file;
    std::vector<std::vector<Sample>> empty;
    
    file.setChannelData(empty, 48000);
    
    EXPECT_TRUE(file.getData().empty());
}

TEST_F(AudioFileTest, GetInvalidChannelData) {
    AudioFile file;
    std::vector<Sample> data = {1.0f, 2.0f, 3.0f};
    file.setData(data, 1, 48000);
    
    auto invalid = file.getChannelData(5);  // Channel doesn't exist
    EXPECT_TRUE(invalid.empty());
}

TEST_F(AudioFileTest, ZeroLengthSineWave) {
    auto file = AudioFile::generateSineWave(440.0f, 0.0, 48000, 0.5f);
    
    EXPECT_TRUE(file.getData().empty());
    EXPECT_EQ(file.getInfo().numSamples, 0);
}

TEST_F(AudioFileTest, CalculateDuration) {
    AudioFile file;
    std::vector<Sample> data(48000);  // 1 second at 48kHz
    file.setData(data, 1, 48000);
    
    EXPECT_NEAR(file.getInfo().durationSeconds, 1.0, 0.001);
}

// ============================================================================
// Sample Rate Conversion (Stub Test)
// ============================================================================

TEST_F(AudioFileTest, SampleRateConversionStub) {
    AudioFile file;
    std::vector<Sample> data = {1.0f, 2.0f, 3.0f};
    file.setData(data, 1, 48000);
    
    // Should return false (not implemented)
    bool result = file.convertSampleRate(44100);
    EXPECT_FALSE(result);
}

// ============================================================================
// Move Semantics Tests
// ============================================================================

TEST_F(AudioFileTest, MoveConstruction) {
    auto file1 = AudioFile::generateSineWave(440.0f, 0.1, 48000, 0.5f);
    size_t originalSize = file1.getData().size();
    
    AudioFile file2(std::move(file1));
    
    EXPECT_EQ(file2.getData().size(), originalSize);
    EXPECT_EQ(file2.getInfo().sampleRate, 48000);
}

TEST_F(AudioFileTest, MoveAssignment) {
    auto file1 = AudioFile::generateSineWave(440.0f, 0.1, 48000, 0.5f);
    size_t originalSize = file1.getData().size();
    
    AudioFile file2;
    file2 = std::move(file1);
    
    EXPECT_EQ(file2.getData().size(), originalSize);
    EXPECT_EQ(file2.getInfo().sampleRate, 48000);
}

// Run all tests
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
