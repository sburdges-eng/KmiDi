/**
 * @file StemExporterTest.cpp
 * @brief Unit tests for StemExporter class
 */

#include <gtest/gtest.h>
#include "daiw/export/StemExporter.h"
#include "daiw/project/ProjectFile.h"
#include <filesystem>

using namespace daiw;
using namespace daiw::export_ns;
using namespace daiw::project;
using namespace daiw::audio;
namespace fs = std::filesystem;

class StemExporterTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create temp directory for test files
        testDir_ = fs::temp_directory_path() / "stem_export_test";
        fs::create_directories(testDir_);
        
        // Create a test project
        project_.setMetadata({"Test Project", "Test Author", "", ""});
        project_.setTempo(120.0f);
        project_.setSampleRate(48000);
    }

    void TearDown() override {
        // Clean up test files
        if (fs::exists(testDir_)) {
            fs::remove_all(testDir_);
        }
    }

    fs::path testDir_;
    ProjectFile project_;
};

// ============================================================================
// Filename Generation Tests
// ============================================================================

TEST_F(StemExporterTest, GenerateStemFilename) {
    auto filename = StemExporter::generateStemFilename(
        "Drums",
        0,
        testDir_.string(),
        AudioFormat::WAV,
        ""
    );
    
    EXPECT_NE(filename.find("Drums"), std::string::npos);
    EXPECT_NE(filename.find(".wav"), std::string::npos);
}

TEST_F(StemExporterTest, GenerateStemFilenameWithSuffix) {
    auto filename = StemExporter::generateStemFilename(
        "Bass",
        1,
        testDir_.string(),
        AudioFormat::WAV,
        "final"
    );
    
    EXPECT_NE(filename.find("Bass"), std::string::npos);
    EXPECT_NE(filename.find("final"), std::string::npos);
    EXPECT_NE(filename.find(".wav"), std::string::npos);
}

TEST_F(StemExporterTest, SanitizeFilename) {
    auto filename = StemExporter::generateStemFilename(
        "Vocals (lead)",
        0,
        testDir_.string(),
        AudioFormat::WAV,
        ""
    );
    
    // Should replace parentheses with underscores
    EXPECT_NE(filename.find("Vocals"), std::string::npos);
    EXPECT_EQ(filename.find("("), std::string::npos);
}

TEST_F(StemExporterTest, EmptyTrackNameFallback) {
    auto filename = StemExporter::generateStemFilename(
        "",
        42,
        testDir_.string(),
        AudioFormat::WAV,
        ""
    );
    
    // Should use Track_N format
    EXPECT_NE(filename.find("Track_42"), std::string::npos);
}

// ============================================================================
// Audio Normalization Tests
// ============================================================================

TEST_F(StemExporterTest, NormalizeAudio) {
    auto file = AudioFile::generateSineWave(440.0f, 0.1, 48000, 0.5f);
    
    // Peak should be around 0.5
    const auto& dataBefore = file.getData();
    float peakBefore = 0.0f;
    for (const auto& sample : dataBefore) {
        peakBefore = std::max(peakBefore, std::abs(sample));
    }
    EXPECT_NEAR(peakBefore, 0.5f, 0.01f);
    
    // Normalize to 0.9
    StemExporter::normalizeAudio(file, 0.9f);
    
    const auto& dataAfter = file.getData();
    float peakAfter = 0.0f;
    for (const auto& sample : dataAfter) {
        peakAfter = std::max(peakAfter, std::abs(sample));
    }
    EXPECT_NEAR(peakAfter, 0.9f, 0.01f);
}

TEST_F(StemExporterTest, NormalizeEmptyAudio) {
    AudioFile file;
    
    // Should not crash
    StemExporter::normalizeAudio(file, 0.9f);
    
    EXPECT_TRUE(file.getData().empty());
}

TEST_F(StemExporterTest, NormalizeSilentAudio) {
    AudioFile file;
    std::vector<Sample> silence(1000, 0.0f);
    file.setData(silence, 1, 48000);
    
    // Should not crash or divide by zero
    StemExporter::normalizeAudio(file, 0.9f);
    
    // Should remain silent
    const auto& data = file.getData();
    for (const auto& sample : data) {
        EXPECT_FLOAT_EQ(sample, 0.0f);
    }
}

// ============================================================================
// MIDI Rendering Tests (Stub)
// ============================================================================

TEST_F(StemExporterTest, RenderMidiTrackStub) {
    Track midiTrack;
    midiTrack.name = "MIDI Track";
    midiTrack.type = TrackType::MIDI;
    
    // Add some MIDI notes
    midi::MidiSequence seq;
    seq.addMessage(midi::MidiMessage::noteOn(0, 60, 100));
    midiTrack.midiSequence = seq;
    
    // Render (should return placeholder sine wave)
    auto audio = StemExporter::renderMidiTrack(midiTrack, 1.0, 48000);
    
    // Should have generated something (stub implementation)
    EXPECT_FALSE(audio.getData().empty());
}

TEST_F(StemExporterTest, RenderEmptyMidiTrack) {
    Track midiTrack;
    midiTrack.name = "Empty MIDI";
    midiTrack.type = TrackType::MIDI;
    
    auto audio = StemExporter::renderMidiTrack(midiTrack, 1.0, 48000);
    
    // Should return empty audio
    EXPECT_TRUE(audio.getData().empty());
}

// ============================================================================
// Single Track Export Tests
// ============================================================================

TEST_F(StemExporterTest, ExportAudioTrack) {
    // Create a test audio file
    auto testAudioFile = testDir_ / "source.wav";
    auto testAudio = AudioFile::generateSineWave(440.0f, 0.1, 48000, 0.5f);
    testAudio.write(testAudioFile.string(), AudioFormat::WAV, SampleFormat::Float32);
    
    // Create audio track
    Track audioTrack;
    audioTrack.name = "Test Audio";
    audioTrack.type = TrackType::Audio;
    audioTrack.audioFilePath = testAudioFile.string();
    audioTrack.volume = 1.0f;
    
    // Export
    StemExporter exporter;
    auto outputFile = testDir_ / "exported.wav";
    auto result = exporter.exportTrack(audioTrack, outputFile.string());
    
    EXPECT_TRUE(result.success) << result.errorMessage;
    EXPECT_TRUE(fs::exists(outputFile));
    EXPECT_EQ(result.trackName, "Test Audio");
}

TEST_F(StemExporterTest, ExportTrackWithVolume) {
    // Create a test audio file
    auto testAudioFile = testDir_ / "source.wav";
    auto testAudio = AudioFile::generateSineWave(440.0f, 0.1, 48000, 1.0f);
    testAudio.write(testAudioFile.string(), AudioFormat::WAV, SampleFormat::Float32);
    
    // Create audio track with 50% volume
    Track audioTrack;
    audioTrack.name = "Half Volume";
    audioTrack.type = TrackType::Audio;
    audioTrack.audioFilePath = testAudioFile.string();
    audioTrack.volume = 0.5f;
    
    // Export
    StemExporter exporter;
    auto outputFile = testDir_ / "half_volume.wav";
    auto result = exporter.exportTrack(audioTrack, outputFile.string());
    
    EXPECT_TRUE(result.success);
    
    // Load and check volume was applied
    AudioFile exported;
    exported.read(outputFile.string());
    
    const auto& data = exported.getData();
    float peak = 0.0f;
    for (const auto& sample : data) {
        peak = std::max(peak, std::abs(sample));
    }
    
    // Peak should be around 0.5 (original 1.0 * volume 0.5)
    EXPECT_NEAR(peak, 0.5f, 0.01f);
}

TEST_F(StemExporterTest, ExportMissingAudioFile) {
    Track audioTrack;
    audioTrack.name = "Missing";
    audioTrack.type = TrackType::Audio;
    audioTrack.audioFilePath = "/nonexistent/file.wav";
    
    StemExporter exporter;
    auto outputFile = testDir_ / "output.wav";
    auto result = exporter.exportTrack(audioTrack, outputFile.string());
    
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.errorMessage.empty());
}

// ============================================================================
// Multi-Track Export Tests
// ============================================================================

TEST_F(StemExporterTest, ExportAllStems) {
    // Create source audio files
    auto audio1File = testDir_ / "drums.wav";
    auto audio1 = AudioFile::generateSineWave(100.0f, 0.1, 48000, 0.5f);
    audio1.write(audio1File.string(), AudioFormat::WAV, SampleFormat::Float32);
    
    auto audio2File = testDir_ / "bass.wav";
    auto audio2 = AudioFile::generateSineWave(200.0f, 0.1, 48000, 0.5f);
    audio2.write(audio2File.string(), AudioFormat::WAV, SampleFormat::Float32);
    
    // Create tracks
    Track track1;
    track1.name = "Drums";
    track1.type = TrackType::Audio;
    track1.audioFilePath = audio1File.string();
    project_.addTrack(track1);
    
    Track track2;
    track2.name = "Bass";
    track2.type = TrackType::Audio;
    track2.audioFilePath = audio2File.string();
    project_.addTrack(track2);
    
    // Export all stems
    StemExporter exporter;
    auto results = exporter.exportAllStems(project_, testDir_.string());
    
    ASSERT_EQ(results.size(), 2);
    EXPECT_TRUE(results[0].success);
    EXPECT_TRUE(results[1].success);
    
    // Check files exist
    EXPECT_TRUE(fs::exists(results[0].filepath));
    EXPECT_TRUE(fs::exists(results[1].filepath));
}

TEST_F(StemExporterTest, ExportSelectedStems) {
    // Create 3 tracks
    for (int i = 0; i < 3; ++i) {
        auto audioFile = testDir_ / ("track" + std::to_string(i) + ".wav");
        auto audio = AudioFile::generateSineWave(100.0f * (i + 1), 0.1, 48000, 0.5f);
        audio.write(audioFile.string(), AudioFormat::WAV, SampleFormat::Float32);
        
        Track track;
        track.name = "Track " + std::to_string(i);
        track.type = TrackType::Audio;
        track.audioFilePath = audioFile.string();
        project_.addTrack(track);
    }
    
    // Export only tracks 0 and 2
    StemExporter exporter;
    std::vector<size_t> indices = {0, 2};
    auto results = exporter.exportSelectedStems(project_, indices, testDir_.string());
    
    ASSERT_EQ(results.size(), 2);
    EXPECT_EQ(results[0].trackName, "Track 0");
    EXPECT_EQ(results[1].trackName, "Track 2");
}

// ============================================================================
// Progress Callback Tests
// ============================================================================

TEST_F(StemExporterTest, ProgressCallback) {
    // Create 3 tracks
    for (int i = 0; i < 3; ++i) {
        auto audioFile = testDir_ / ("track" + std::to_string(i) + ".wav");
        auto audio = AudioFile::generateSineWave(100.0f, 0.05, 48000, 0.5f);
        audio.write(audioFile.string(), AudioFormat::WAV, SampleFormat::Float32);
        
        Track track;
        track.name = "Track " + std::to_string(i);
        track.type = TrackType::Audio;
        track.audioFilePath = audioFile.string();
        project_.addTrack(track);
    }
    
    // Track progress
    int callbackCount = 0;
    size_t lastTrack = 0;
    size_t totalTracks = 0;
    
    StemExporter exporter;
    exporter.setProgressCallback([&](size_t current, size_t total, const std::string& name) {
        callbackCount++;
        lastTrack = current;
        totalTracks = total;
    });
    
    exporter.exportAllStems(project_, testDir_.string());
    
    EXPECT_EQ(callbackCount, 3);
    EXPECT_EQ(lastTrack, 2);  // Last track index
    EXPECT_EQ(totalTracks, 3);
}

// ============================================================================
// Export Options Tests
// ============================================================================

TEST_F(StemExporterTest, ExportWithNormalization) {
    // Create audio file with low volume
    auto audioFile = testDir_ / "quiet.wav";
    auto audio = AudioFile::generateSineWave(440.0f, 0.1, 48000, 0.2f);
    audio.write(audioFile.string(), AudioFormat::WAV, SampleFormat::Float32);
    
    Track track;
    track.name = "Quiet Track";
    track.type = TrackType::Audio;
    track.audioFilePath = audioFile.string();
    
    // Export with normalization
    ExportOptions options;
    options.normalizeStems = true;
    
    StemExporter exporter;
    auto outputFile = testDir_ / "normalized.wav";
    auto result = exporter.exportTrack(track, outputFile.string(), options);
    
    EXPECT_TRUE(result.success);
    
    // Load and check normalization
    AudioFile normalized;
    normalized.read(outputFile.string());
    
    const auto& data = normalized.getData();
    float peak = 0.0f;
    for (const auto& sample : data) {
        peak = std::max(peak, std::abs(sample));
    }
    
    // Should be normalized to 0.9 (default target)
    EXPECT_NEAR(peak, 0.9f, 0.01f);
}

// Run all tests
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
