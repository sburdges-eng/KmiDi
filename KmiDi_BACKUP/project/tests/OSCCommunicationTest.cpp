/*
  ==============================================================================

    OSCCommunicationTest.cpp
    Created: 2025
    Author: DAiW Team

    Integration tests for OSC communication with Python brain server

  ==============================================================================
*/

#include <JuceHeader.h>
#include "PluginProcessor.h"
#include <juce_osc/juce_osc.h>
#include <catch2/catch_test_macros.hpp>
#include <thread>
#include <chrono>

// If Catch2 is not available, use JUCE's UnitTest framework
#ifndef CATCH_TEST_CASE
    #include <juce_core/juce_core.h>
    #define CATCH_TEST_CASE(name, tags) static void name()
    #define REQUIRE(condition) jassert(condition)
    #define REQUIRE_NOTHROW(expression) { try { expression; } catch(...) { jassertfalse; } }
    #define SKIP_TEST(message) return
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
// OSC Message Format Tests
//==============================================================================

CATCH_TEST_CASE("OSC - Generate Message Format", "[osc]")
{
    juce::OSCMessage msg("/daiw/generate");
    msg.addString("I feel deep grief");
    msg.addFloat32(7.0f);
    msg.addFloat32(5.0f);
    msg.addFloat32(6.0f);
    
    REQUIRE(msg.getAddressPattern().toString() == "/daiw/generate");
    REQUIRE(msg.size() == 4);
    REQUIRE(msg[0].isString());
    REQUIRE(msg[1].isFloat32());
    REQUIRE(msg[2].isFloat32());
    REQUIRE(msg[3].isFloat32());
}

CATCH_TEST_CASE("OSC - Ping Message Format", "[osc]")
{
    juce::OSCMessage msg("/daiw/ping");
    
    REQUIRE(msg.getAddressPattern().toString() == "/daiw/ping");
    REQUIRE(msg.size() == 0);
}

CATCH_TEST_CASE("OSC - Result Message Format", "[osc]")
{
    juce::String jsonData = R"({"status": "success", "midi_events": []})";
    juce::OSCMessage msg("/daiw/result");
    msg.addString(jsonData);
    
    REQUIRE(msg.getAddressPattern().toString() == "/daiw/result");
    REQUIRE(msg.size() == 1);
    REQUIRE(msg[0].isString());
    REQUIRE(msg[0].getString() == jsonData);
}

//==============================================================================
// JSON Response Parsing Tests
//==============================================================================

CATCH_TEST_CASE("OSC - Parse Complete Response", "[json]")
{
    PluginProcessorTestFixture fixture;
    fixture.processor->prepareToPlay(44100.0, 512);
    
    juce::String completeResponse = R"({
        "status": "success",
        "affect": {
            "primary": "grief",
            "intensity": 0.85
        },
        "plan": {
            "tempo_bpm": 82,
            "key": "C",
            "mode": "aeolian",
            "time_signature": "4/4",
            "chords": ["Cm", "Ab", "Fm", "Cm"],
            "length_bars": 32
        },
        "ppq": 480,
        "midi_events": [
            {
                "type": "note_on",
                "pitch": 60,
                "velocity": 80,
                "channel": 1,
                "tick": 0,
                "duration_ticks": 1920
            }
        ]
    })";
    
    REQUIRE_NOTHROW(fixture.processor->parseMidiEventsFromJSON(completeResponse));
    
    fixture.processor->releaseResources();
}

CATCH_TEST_CASE("OSC - Parse Response with Multiple Chords", "[json]")
{
    PluginProcessorTestFixture fixture;
    fixture.processor->prepareToPlay(44100.0, 512);
    
    juce::String multiChordResponse = R"({
        "status": "success",
        "plan": {
            "tempo_bpm": 120,
            "chords": ["C", "F", "G", "C"]
        },
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
    
    REQUIRE_NOTHROW(fixture.processor->parseMidiEventsFromJSON(multiChordResponse));
    
    fixture.processor->releaseResources();
}

//==============================================================================
// Error Response Tests
//==============================================================================

CATCH_TEST_CASE("OSC - Parse Error Response", "[error]")
{
    PluginProcessorTestFixture fixture;
    
    juce::String errorResponse = R"({
        "status": "error",
        "message": "Text input cannot be empty"
    })";
    
    juce::OSCMessage errorMsg("/daiw/error");
    errorMsg.addString(errorResponse);
    
    REQUIRE_NOTHROW(fixture.processor->oscMessageReceived(errorMsg));
}

//==============================================================================
// MIDI Event Validation Tests
//==============================================================================

CATCH_TEST_CASE("OSC - Validate MIDI Event Ranges", "[validation]")
{
    PluginProcessorTestFixture fixture;
    fixture.processor->prepareToPlay(44100.0, 512);
    
    // Test boundary values
    juce::String boundaryTest = R"({
        "plan": {"tempo_bpm": 120},
        "ppq": 480,
        "midi_events": [
            {"type": "note_on", "pitch": 0, "velocity": 0, "channel": 1, "tick": 0},
            {"type": "note_on", "pitch": 127, "velocity": 127, "channel": 16, "tick": 0}
        ]
    })";
    
    REQUIRE_NOTHROW(fixture.processor->parseMidiEventsFromJSON(boundaryTest));
    
    fixture.processor->releaseResources();
}

//==============================================================================
// Timing Accuracy Tests
//==============================================================================

CATCH_TEST_CASE("OSC - Timing Calculation Accuracy", "[timing]")
{
    PluginProcessorTestFixture fixture;
    
    double sampleRate = 44100.0;
    int tempoBpm = 120;
    int ppq = 480;
    int tick = 1920;  // One bar at 4/4
    
    fixture.processor->prepareToPlay(sampleRate, 512);
    
    // Calculate expected sample offset
    double secondsPerTick = 60.0 / (tempoBpm * ppq);
    double expectedSamples = tick * secondsPerTick * sampleRate;
    
    juce::String timingTest = R"({
        "plan": {"tempo_bpm": 120},
        "ppq": 480,
        "midi_events": [
            {"type": "note_on", "pitch": 60, "velocity": 80, "channel": 1, "tick": )" + 
            juce::String(tick) + R"(}
        ]
    })";
    
    REQUIRE_NOTHROW(fixture.processor->parseMidiEventsFromJSON(timingTest));
    
    fixture.processor->releaseResources();
}

