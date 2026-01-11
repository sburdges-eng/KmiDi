/**
 * Test suite for Penta-Core Mixer Engine
 *
 * Tests RT-safe mixer functionality including:
 * - Channel strip processing (gain, pan, mute, solo)
 * - Send/return buses
 * - Master bus with limiting
 * - Metering
 * - Solo logic
 */

#include "penta/mixer/MixerEngine.h"
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

using namespace penta::mixer;

// Test fixture for mixer tests
class MixerEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        mixer = std::make_unique<MixerEngine>(48000.0);
        mixer->setNumChannels(4);
        mixer->setNumSendBuses(2);
    }

    void TearDown() override {
        mixer.reset();
    }

    // Helper: Generate test audio (sine wave)
    std::vector<float> generateSine(float frequency, size_t numFrames, float sampleRate = 48000.0f) {
        std::vector<float> buffer(numFrames);
        for (size_t i = 0; i < numFrames; ++i) {
            float t = static_cast<float>(i) / sampleRate;
            buffer[i] = std::sin(2.0f * 3.14159265359f * frequency * t);
        }
        return buffer;
    }

    // Helper: Check if buffer is silent
    bool isSilent(const float* buffer, size_t numFrames, float threshold = 0.0001f) {
        for (size_t i = 0; i < numFrames; ++i) {
            if (std::abs(buffer[i]) > threshold) {
                return false;
            }
        }
        return true;
    }

    std::unique_ptr<MixerEngine> mixer;
};

// =============================================================================
// Basic Functionality Tests
// =============================================================================

TEST_F(MixerEngineTest, InitialState) {
    EXPECT_EQ(mixer->getNumChannels(), 4);
    EXPECT_EQ(mixer->getNumSendBuses(), 2);

    // Check default values
    for (size_t ch = 0; ch < 4; ++ch) {
        EXPECT_FLOAT_EQ(mixer->getChannelPeak(ch), 0.0f);
        EXPECT_FLOAT_EQ(mixer->getChannelRMS(ch), 0.0f);
    }

    EXPECT_FLOAT_EQ(mixer->getMasterPeakL(), 0.0f);
    EXPECT_FLOAT_EQ(mixer->getMasterPeakR(), 0.0f);
}

TEST_F(MixerEngineTest, ChannelGain) {
    const size_t numFrames = 512;
    auto sine = generateSine(440.0f, numFrames);

    // Create input/output buffers
    const float* inputs[4] = {sine.data(), nullptr, nullptr, nullptr};
    std::vector<float> outL(numFrames);
    std::vector<float> outR(numFrames);
    float* outputs[2] = {outL.data(), outR.data()};

    // Test unity gain (0 dB)
    mixer->setChannelGain(0, 0.0f);
    mixer->processAudio(inputs, outputs, numFrames);

    // Check output is present
    EXPECT_FALSE(isSilent(outL.data(), numFrames));
    EXPECT_FALSE(isSilent(outR.data(), numFrames));

    // Test -inf gain (mute via gain)
    mixer->setChannelGain(0, -60.0f);
    mixer->processAudio(inputs, outputs, numFrames);

    // Output should be very quiet
    float maxLevel = 0.0f;
    for (size_t i = 0; i < numFrames; ++i) {
        maxLevel = std::max(maxLevel, std::abs(outL[i]));
    }
    EXPECT_LT(maxLevel, 0.01f);  // Should be significantly attenuated
}

TEST_F(MixerEngineTest, ChannelPan) {
    const size_t numFrames = 512;
    auto sine = generateSine(440.0f, numFrames);

    const float* inputs[4] = {sine.data(), nullptr, nullptr, nullptr};
    std::vector<float> outL(numFrames);
    std::vector<float> outR(numFrames);
    float* outputs[2] = {outL.data(), outR.data()};

    // Test hard left (-1.0)
    mixer->setChannelPan(0, -1.0f);
    mixer->processAudio(inputs, outputs, numFrames);

    float peakL = 0.0f, peakR = 0.0f;
    for (size_t i = 0; i < numFrames; ++i) {
        peakL = std::max(peakL, std::abs(outL[i]));
        peakR = std::max(peakR, std::abs(outR[i]));
    }

    // Left should be louder than right
    EXPECT_GT(peakL, peakR);

    // Test hard right (+1.0)
    mixer->setChannelPan(0, 1.0f);
    mixer->processAudio(inputs, outputs, numFrames);

    peakL = peakR = 0.0f;
    for (size_t i = 0; i < numFrames; ++i) {
        peakL = std::max(peakL, std::abs(outL[i]));
        peakR = std::max(peakR, std::abs(outR[i]));
    }

    // Right should be louder than left
    EXPECT_GT(peakR, peakL);

    // Test center (0.0) - should be equal power
    mixer->setChannelPan(0, 0.0f);
    mixer->processAudio(inputs, outputs, numFrames);

    peakL = peakR = 0.0f;
    for (size_t i = 0; i < numFrames; ++i) {
        peakL = std::max(peakL, std::abs(outL[i]));
        peakR = std::max(peakR, std::abs(outR[i]));
    }

    // Should be roughly equal (constant power pan law)
    EXPECT_NEAR(peakL, peakR, 0.1f);
}

TEST_F(MixerEngineTest, ChannelMute) {
    const size_t numFrames = 512;
    auto sine = generateSine(440.0f, numFrames);

    const float* inputs[4] = {sine.data(), nullptr, nullptr, nullptr};
    std::vector<float> outL(numFrames);
    std::vector<float> outR(numFrames);
    float* outputs[2] = {outL.data(), outR.data()};

    // Unmuted - should have output
    mixer->setChannelMute(0, false);
    mixer->processAudio(inputs, outputs, numFrames);
    EXPECT_FALSE(isSilent(outL.data(), numFrames));

    // Muted - should be silent
    mixer->setChannelMute(0, true);
    mixer->processAudio(inputs, outputs, numFrames);
    EXPECT_TRUE(isSilent(outL.data(), numFrames));
}

TEST_F(MixerEngineTest, ChannelSolo) {
    const size_t numFrames = 512;
    auto sine1 = generateSine(440.0f, numFrames);
    auto sine2 = generateSine(880.0f, numFrames);

    const float* inputs[4] = {sine1.data(), sine2.data(), nullptr, nullptr};
    std::vector<float> outL(numFrames);
    std::vector<float> outR(numFrames);
    float* outputs[2] = {outL.data(), outR.data()};

    // No solo - both channels should be heard
    mixer->processAudio(inputs, outputs, numFrames);
    float peakNoSolo = 0.0f;
    for (size_t i = 0; i < numFrames; ++i) {
        peakNoSolo = std::max(peakNoSolo, std::abs(outL[i]));
    }
    EXPECT_GT(peakNoSolo, 0.0f);

    // Solo channel 0 - only channel 0 should be heard
    mixer->setChannelSolo(0, true);
    EXPECT_TRUE(mixer->isAnySoloed());

    mixer->processAudio(inputs, outputs, numFrames);

    // Output should still be present (channel 0 is soloed)
    EXPECT_FALSE(isSilent(outL.data(), numFrames));

    // Clear solo
    mixer->clearAllSolo();
    EXPECT_FALSE(mixer->isAnySoloed());
}

// =============================================================================
// Send/Return Tests
// =============================================================================

TEST_F(MixerEngineTest, SendBus) {
    const size_t numFrames = 512;
    auto sine = generateSine(440.0f, numFrames);

    const float* inputs[4] = {sine.data(), nullptr, nullptr, nullptr};
    std::vector<float> outL(numFrames);
    std::vector<float> outR(numFrames);
    float* outputs[2] = {outL.data(), outR.data()};

    // Set send level for channel 0 to send bus 0
    mixer->setChannelSend(0, 0, 0.5f);
    mixer->setSendReturnLevel(0, 1.0f);

    mixer->processAudio(inputs, outputs, numFrames);

    // Output should contain both dry signal and send return
    EXPECT_FALSE(isSilent(outL.data(), numFrames));
}

// =============================================================================
// Master Bus Tests
// =============================================================================

TEST_F(MixerEngineTest, MasterGain) {
    const size_t numFrames = 512;
    auto sine = generateSine(440.0f, numFrames);

    const float* inputs[4] = {sine.data(), nullptr, nullptr, nullptr};
    std::vector<float> outL(numFrames);
    std::vector<float> outR(numFrames);
    float* outputs[2] = {outL.data(), outR.data()};

    // Unity gain
    mixer->setMasterGain(0.0f);
    mixer->processAudio(inputs, outputs, numFrames);

    float peak1 = 0.0f;
    for (size_t i = 0; i < numFrames; ++i) {
        peak1 = std::max(peak1, std::abs(outL[i]));
    }

    // +6 dB gain (approximately 2x amplitude)
    mixer->setMasterGain(6.0f);
    mixer->processAudio(inputs, outputs, numFrames);

    float peak2 = 0.0f;
    for (size_t i = 0; i < numFrames; ++i) {
        peak2 = std::max(peak2, std::abs(outL[i]));
    }

    // Peak should roughly double
    EXPECT_GT(peak2, peak1 * 1.5f);
}

TEST_F(MixerEngineTest, MasterLimiter) {
    const size_t numFrames = 512;

    // Generate loud signal
    std::vector<float> loudSine(numFrames);
    for (size_t i = 0; i < numFrames; ++i) {
        loudSine[i] = 2.0f * std::sin(2.0f * 3.14159265359f * 440.0f * i / 48000.0f);
    }

    const float* inputs[4] = {loudSine.data(), nullptr, nullptr, nullptr};
    std::vector<float> outL(numFrames);
    std::vector<float> outR(numFrames);
    float* outputs[2] = {outL.data(), outR.data()};

    // Enable limiter with threshold at -1 dB
    mixer->setMasterLimiter(true, -1.0f);
    mixer->processAudio(inputs, outputs, numFrames);

    // Check that output doesn't exceed threshold
    float peak = 0.0f;
    for (size_t i = 0; i < numFrames; ++i) {
        peak = std::max(peak, std::abs(outL[i]));
    }

    // Should be limited (within reasonable tolerance)
    const float thresholdLinear = std::pow(10.0f, -1.0f / 20.0f);  // -1 dB
    EXPECT_LT(peak, thresholdLinear * 1.2f);  // Allow some tolerance for attack time
}

// =============================================================================
// Metering Tests
// =============================================================================

TEST_F(MixerEngineTest, ChannelMetering) {
    const size_t numFrames = 512;
    auto sine = generateSine(440.0f, numFrames);

    const float* inputs[4] = {sine.data(), nullptr, nullptr, nullptr};
    std::vector<float> outL(numFrames);
    std::vector<float> outR(numFrames);
    float* outputs[2] = {outL.data(), outR.data()};

    mixer->processAudio(inputs, outputs, numFrames);

    // Check that meters updated
    EXPECT_GT(mixer->getChannelPeak(0), 0.0f);
    EXPECT_GT(mixer->getChannelRMS(0), 0.0f);

    // RMS should be less than peak for a sine wave
    EXPECT_LT(mixer->getChannelRMS(0), mixer->getChannelPeak(0));

    // Reset meters
    mixer->resetAllMeters();
    EXPECT_FLOAT_EQ(mixer->getChannelPeak(0), 0.0f);
    EXPECT_FLOAT_EQ(mixer->getChannelRMS(0), 0.0f);
}

// =============================================================================
// Multi-channel Mixing Tests
// =============================================================================

TEST_F(MixerEngineTest, MultiChannelMix) {
    const size_t numFrames = 512;

    // Generate different frequencies for each channel
    auto sine1 = generateSine(440.0f, numFrames);
    auto sine2 = generateSine(880.0f, numFrames);
    auto sine3 = generateSine(1320.0f, numFrames);
    auto sine4 = generateSine(1760.0f, numFrames);

    const float* inputs[4] = {sine1.data(), sine2.data(), sine3.data(), sine4.data()};
    std::vector<float> outL(numFrames);
    std::vector<float> outR(numFrames);
    float* outputs[2] = {outL.data(), outR.data()};

    // Process with all channels active
    mixer->processAudio(inputs, outputs, numFrames);

    // All channels should contribute
    for (size_t ch = 0; ch < 4; ++ch) {
        EXPECT_GT(mixer->getChannelPeak(ch), 0.0f);
    }

    // Master should have output
    EXPECT_GT(mixer->getMasterPeakL(), 0.0f);
    EXPECT_GT(mixer->getMasterPeakR(), 0.0f);
}

// =============================================================================
// Performance Tests
// =============================================================================

TEST_F(MixerEngineTest, ProcessingPerformance) {
    const size_t numFrames = 512;
    const size_t numIterations = 1000;

    // Generate test signals
    std::vector<std::vector<float>> testSignals(4);
    for (size_t ch = 0; ch < 4; ++ch) {
        testSignals[ch] = generateSine(440.0f * (ch + 1), numFrames);
    }

    const float* inputs[4] = {
        testSignals[0].data(),
        testSignals[1].data(),
        testSignals[2].data(),
        testSignals[3].data()
    };

    std::vector<float> outL(numFrames);
    std::vector<float> outR(numFrames);
    float* outputs[2] = {outL.data(), outR.data()};

    // Time processing
    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < numIterations; ++i) {
        mixer->processAudio(inputs, outputs, numFrames);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Calculate average time per buffer
    double avgTime = duration.count() / static_cast<double>(numIterations);

    // Should be fast enough for RT (512 samples @ 48kHz = ~10.67ms available)
    // We want processing to be well under 1ms per buffer
    EXPECT_LT(avgTime, 1000.0);  // Less than 1ms per buffer

    std::cout << "Average processing time per buffer: " << avgTime << " μs" << std::endl;
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(MixerEngineTest, SilentInput) {
    const size_t numFrames = 512;
    std::vector<float> silence(numFrames, 0.0f);

    const float* inputs[4] = {silence.data(), nullptr, nullptr, nullptr};
    std::vector<float> outL(numFrames);
    std::vector<float> outR(numFrames);
    float* outputs[2] = {outL.data(), outR.data()};

    mixer->processAudio(inputs, outputs, numFrames);

    // Output should be silent
    EXPECT_TRUE(isSilent(outL.data(), numFrames));
    EXPECT_TRUE(isSilent(outR.data(), numFrames));
}

TEST_F(MixerEngineTest, AllChannelsMuted) {
    const size_t numFrames = 512;
    auto sine = generateSine(440.0f, numFrames);

    const float* inputs[4] = {sine.data(), sine.data(), sine.data(), sine.data()};
    std::vector<float> outL(numFrames);
    std::vector<float> outR(numFrames);
    float* outputs[2] = {outL.data(), outR.data()};

    // Mute all channels
    for (size_t ch = 0; ch < 4; ++ch) {
        mixer->setChannelMute(ch, true);
    }

    mixer->processAudio(inputs, outputs, numFrames);

    // Output should be silent
    EXPECT_TRUE(isSilent(outL.data(), numFrames));
    EXPECT_TRUE(isSilent(outR.data(), numFrames));
}

// Main test runner
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
