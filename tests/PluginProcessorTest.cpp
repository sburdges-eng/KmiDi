/*
  ==============================================================================

    PluginProcessorTest.cpp
    Created: 2025
    Author: DAiW Team

    Unit tests for DAiW Bridge Plugin Processor

  ==============================================================================
*/

#include <JuceHeader.h>
#include "PluginProcessor.h"

// Use Catch2 if available, otherwise fall back to JUCE UnitTest
#ifdef CATCH_TEST_CASE
    #include <catch2/catch_test_macros.hpp>
    #include <catch2/matchers/catch_matchers.hpp>
#else
    #include <juce_core/juce_core.h>
    #define CATCH_TEST_CASE(name, tags) static void name()
    #define REQUIRE(condition) jassert(condition)
    #define REQUIRE_NOTHROW(expression) { try { expression; } catch(...) { jassertfalse; } }
    #define REQUIRE_FALSE(condition) jassert(!(condition))
    #define REQUIRE_THROWS(expression) { bool threw = false; try { expression; } catch(...) { threw = true; } jassert(threw); }
    #define REQUIRE_THAT(value, matcher) jassert(matcher(value))
#endif

// If Catch2 is not available, use JUCE's UnitTest framework
#ifndef CATCH_TEST_CASE
    #include <juce_core/juce_core.h>
    #define CATCH_TEST_CASE(name, tags) static void name()
    #define REQUIRE(condition) jassert(condition)
    #define REQUIRE_THROWS(expression) { bool threw = false; try { expression; } catch(...) { threw = true; } jassert(threw); }
    #define REQUIRE_FALSE(condition) jassert(!(condition))
    #define REQUIRE_THAT(value, matcher) jassert(matcher(value))
#endif

//==============================================================================
// Test Fixtures
//==============================================================================

class PluginProcessorTestFixture
{
public:
    PluginProcessorTestFixture()
    {
        processor = std::make_unique<DAiWBridgeAudioProcessor>();
    }
    
    ~PluginProcessorTestFixture()
    {
        processor.reset();
    }
    
    std::unique_ptr<DAiWBridgeAudioProcessor> processor;
};

//==============================================================================
// Basic Functionality Tests
//==============================================================================

CATCH_TEST_CASE("PluginProcessor - Initialization", "[processor]")
{
    PluginProcessorTestFixture fixture;
    
    REQUIRE(fixture.processor != nullptr);
    REQUIRE(fixture.processor->getName() == JucePlugin_Name);
    REQUIRE(fixture.processor->acceptsMidi() == true);
    REQUIRE(fixture.processor->producesMidi() == true);
    REQUIRE(fixture.processor->isMidiEffect() == true);
}

CATCH_TEST_CASE("PluginProcessor - Connection Status", "[processor]")
{
    PluginProcessorTestFixture fixture;
    
    // Initially should be false (connection confirmed by pong)
    REQUIRE_FALSE(fixture.processor->isConnected());
    
    // After ping, connection status may change (depends on server)
    fixture.processor->sendPing();
    // Note: Connection status will only be true after receiving pong
}

CATCH_TEST_CASE("PluginProcessor - Prepare to Play", "[processor]")
{
    PluginProcessorTestFixture fixture;
    
    double sampleRate = 44100.0;
    int samplesPerBlock = 512;
    
    // Should not throw
    REQUIRE_NOTHROW(fixture.processor->prepareToPlay(sampleRate, samplesPerBlock));
    
    fixture.processor->releaseResources();
}

CATCH_TEST_CASE("PluginProcessor - Process Block", "[processor]")
{
    PluginProcessorTestFixture fixture;
    
    double sampleRate = 44100.0;
    int samplesPerBlock = 512;
    int numChannels = 2;
    
    fixture.processor->prepareToPlay(sampleRate, samplesPerBlock);
    
    juce::AudioBuffer<float> buffer(numChannels, samplesPerBlock);
    juce::MidiBuffer midiMessages;
    
    // Clear buffer
    buffer.clear();
    
    // Process should not throw
    REQUIRE_NOTHROW(fixture.processor->processBlock(buffer, midiMessages));
    
    fixture.processor->releaseResources();
}

//==============================================================================
// OSC Communication Tests
//==============================================================================

CATCH_TEST_CASE("PluginProcessor - Send Generate Request", "[osc]")
{
    PluginProcessorTestFixture fixture;
    
    juce::String text = "I feel deep grief";
    float motivation = 7.0f;
    float chaos = 5.0f;
    float vulnerability = 6.0f;
    
    // Should not throw even if not connected
    REQUIRE_NOTHROW(
        fixture.processor->sendGenerateRequest(text, motivation, chaos, vulnerability)
    );
}

CATCH_TEST_CASE("PluginProcessor - Send Ping", "[osc]")
{
    PluginProcessorTestFixture fixture;
    
    // Should not throw
    REQUIRE_NOTHROW(fixture.processor->sendPing());
}

//==============================================================================
// JSON Parsing Tests
//==============================================================================

CATCH_TEST_CASE("PluginProcessor - Parse Valid MIDI JSON", "[json]")
{
    PluginProcessorTestFixture fixture;
    
    // Valid JSON with MIDI events
    juce::String validJson = R"({
        "status": "success",
        "plan": {
            "tempo_bpm": 120
        },
        "ppq": 480,
        "midi_events": [
            {
                "type": "note_on",
                "pitch": 60,
                "velocity": 80,
                "channel": 1,
                "tick": 0
            },
            {
                "type": "note_off",
                "pitch": 60,
                "velocity": 0,
                "channel": 1,
                "tick": 1920
            }
        ]
    })";
    
    // Parse should not throw
    REQUIRE_NOTHROW(fixture.processor->parseMidiEventsFromJSON(validJson));
}

CATCH_TEST_CASE("PluginProcessor - Parse Invalid JSON", "[json]")
{
    PluginProcessorTestFixture fixture;
    
    juce::String invalidJson = "{ invalid json }";
    
    // Should handle gracefully without throwing
    REQUIRE_NOTHROW(fixture.processor->parseMidiEventsFromJSON(invalidJson));
}

CATCH_TEST_CASE("PluginProcessor - Parse Empty JSON", "[json]")
{
    PluginProcessorTestFixture fixture;
    
    juce::String emptyJson = "{}";
    
    // Should handle gracefully
    REQUIRE_NOTHROW(fixture.processor->parseMidiEventsFromJSON(emptyJson));
}

CATCH_TEST_CASE("PluginProcessor - Parse JSON with Missing Properties", "[json]")
{
    PluginProcessorTestFixture fixture;
    
    // JSON missing some properties
    juce::String partialJson = R"({
        "midi_events": [
            {
                "type": "note_on",
                "pitch": 60
            }
        ]
    })";
    
    // Should handle missing properties gracefully
    REQUIRE_NOTHROW(fixture.processor->parseMidiEventsFromJSON(partialJson));
}

CATCH_TEST_CASE("PluginProcessor - Parse JSON with Invalid Pitch", "[json]")
{
    PluginProcessorTestFixture fixture;
    
    juce::String invalidPitchJson = R"({
        "midi_events": [
            {
                "type": "note_on",
                "pitch": 200,
                "velocity": 80,
                "channel": 1,
                "tick": 0
            }
        ]
    })";
    
    // Should handle invalid pitch (out of range) gracefully
    REQUIRE_NOTHROW(fixture.processor->parseMidiEventsFromJSON(invalidPitchJson));
}

//==============================================================================
// MIDI Event Tests
//==============================================================================

CATCH_TEST_CASE("PluginProcessor - MIDI Event Clamping", "[midi]")
{
    PluginProcessorTestFixture fixture;
    
    fixture.processor->prepareToPlay(44100.0, 512);
    
    // Test with various invalid values that should be clamped
    juce::String testJson = R"({
        "plan": {"tempo_bpm": 120},
        "ppq": 480,
        "midi_events": [
            {
                "type": "note_on",
                "pitch": -10,
                "velocity": 200,
                "channel": 20,
                "tick": 0
            }
        ]
    })";
    
    // Should handle and clamp values
    REQUIRE_NOTHROW(fixture.processor->parseMidiEventsFromJSON(testJson));
    
    fixture.processor->releaseResources();
}

CATCH_TEST_CASE("PluginProcessor - Multiple MIDI Events", "[midi]")
{
    PluginProcessorTestFixture fixture;
    
    fixture.processor->prepareToPlay(44100.0, 512);
    
    // JSON with multiple events
    juce::String multiEventJson = R"({
        "plan": {"tempo_bpm": 120},
        "ppq": 480,
        "midi_events": [
            {"type": "note_on", "pitch": 60, "velocity": 80, "channel": 1, "tick": 0},
            {"type": "note_on", "pitch": 64, "velocity": 80, "channel": 1, "tick": 0},
            {"type": "note_on", "pitch": 67, "velocity": 80, "channel": 1, "tick": 0},
            {"type": "note_off", "pitch": 60, "velocity": 0, "channel": 1, "tick": 1920},
            {"type": "note_off", "pitch": 64, "velocity": 0, "channel": 1, "tick": 1920},
            {"type": "note_off", "pitch": 67, "velocity": 0, "channel": 1, "tick": 1920}
        ]
    })";
    
    REQUIRE_NOTHROW(fixture.processor->parseMidiEventsFromJSON(multiEventJson));
    
    fixture.processor->releaseResources();
}

//==============================================================================
// Timing Tests
//==============================================================================

CATCH_TEST_CASE("PluginProcessor - Tempo-Aware Timing", "[timing]")
{
    PluginProcessorTestFixture fixture;
    
    double sampleRate = 44100.0;
    fixture.processor->prepareToPlay(sampleRate, 512);
    
    // Test with different tempos
    juce::String slowTempoJson = R"({
        "plan": {"tempo_bpm": 60},
        "ppq": 480,
        "midi_events": [
            {"type": "note_on", "pitch": 60, "velocity": 80, "channel": 1, "tick": 0}
        ]
    })";
    
    REQUIRE_NOTHROW(fixture.processor->parseMidiEventsFromJSON(slowTempoJson));
    
    juce::String fastTempoJson = R"({
        "plan": {"tempo_bpm": 180},
        "ppq": 480,
        "midi_events": [
            {"type": "note_on", "pitch": 60, "velocity": 80, "channel": 1, "tick": 0}
        ]
    })";
    
    REQUIRE_NOTHROW(fixture.processor->parseMidiEventsFromJSON(fastTempoJson));
    
    fixture.processor->releaseResources();
}

//==============================================================================
// Error Handling Tests
//==============================================================================

CATCH_TEST_CASE("PluginProcessor - Handle OSC Error Message", "[error]")
{
    PluginProcessorTestFixture fixture;
    
    // Simulate error message from brain server
    juce::OSCMessage errorMsg("/daiw/error");
    errorMsg.addString(R"({"status": "error", "message": "Test error"})");
    
    // Should handle error gracefully
    REQUIRE_NOTHROW(fixture.processor->oscMessageReceived(errorMsg));
}

CATCH_TEST_CASE("PluginProcessor - Handle Unknown OSC Message", "[error]")
{
    PluginProcessorTestFixture fixture;
    
    juce::OSCMessage unknownMsg("/daiw/unknown");
    unknownMsg.addString("test");
    
    // Should handle unknown messages gracefully
    REQUIRE_NOTHROW(fixture.processor->oscMessageReceived(unknownMsg));
}

//==============================================================================
// Editor Tests
//==============================================================================

CATCH_TEST_CASE("PluginProcessor - Create Editor", "[editor]")
{
    PluginProcessorTestFixture fixture;
    
    REQUIRE(fixture.processor->hasEditor() == true);
    
    auto* editor = fixture.processor->createEditor();
    REQUIRE(editor != nullptr);
    
    delete editor;
}

//==============================================================================
// State Management Tests
//==============================================================================

CATCH_TEST_CASE("PluginProcessor - State Save/Load", "[state]")
{
    PluginProcessorTestFixture fixture;
    
    juce::MemoryBlock stateData;
    
    // Save state
    REQUIRE_NOTHROW(fixture.processor->getStateInformation(stateData));
    
    // Load state
    REQUIRE_NOTHROW(fixture.processor->setStateInformation(
        stateData.getData(),
        (int)stateData.getSize()
    ));
}

//==============================================================================
// Performance Tests
//==============================================================================

CATCH_TEST_CASE("PluginProcessor - Process Block Performance", "[performance]")
{
    PluginProcessorTestFixture fixture;
    
    double sampleRate = 44100.0;
    int samplesPerBlock = 512;
    int numChannels = 2;
    int numBlocks = 100;
    
    fixture.processor->prepareToPlay(sampleRate, samplesPerBlock);
    
    juce::AudioBuffer<float> buffer(numChannels, samplesPerBlock);
    juce::MidiBuffer midiMessages;
    
    auto startTime = juce::Time::getMillisecondCounterHiRes();
    
    for (int i = 0; i < numBlocks; ++i)
    {
        buffer.clear();
        fixture.processor->processBlock(buffer, midiMessages);
    }
    
    auto endTime = juce::Time::getMillisecondCounterHiRes();
    auto elapsed = endTime - startTime;
    
    // Should process 100 blocks in reasonable time (< 100ms for 512 samples at 44.1kHz)
    REQUIRE(elapsed < 100.0);
    
    fixture.processor->releaseResources();
}

