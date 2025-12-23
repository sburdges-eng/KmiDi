/*
  ==============================================================================

    PluginProcessor.h
    Created: 2025
    Author: DAiW Team

    DAiW Bridge Plugin - Connects DAW to Python Brain via OSC

  ==============================================================================
*/

#pragma once

#include <JuceHeader.h>
#include <juce_osc/juce_osc.h>

//==============================================================================
/**
    DAiW Bridge Audio Processor

    This plugin acts as a bridge between your DAW and the Python DAiW brain.
    It sends OSC messages to the brain server and receives MIDI events back.
*/
class DAiWBridgeAudioProcessor : public juce::AudioProcessor,
                                  private juce::OSCReceiver::Listener<juce::OSCReceiver::MessageLoopCallback>
{
public:
    //==============================================================================
    DAiWBridgeAudioProcessor();
    ~DAiWBridgeAudioProcessor() override;

    //==============================================================================
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

   #ifndef JucePlugin_PreferredChannelConfigurations
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
   #endif

    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    //==============================================================================
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    //==============================================================================
    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    //==============================================================================
    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram (int index) override;
    const juce::String getProgramName (int index) override;
    void changeProgramName (int index, const juce::String& newName) override;

    //==============================================================================
    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    //==============================================================================
    // DAiW-specific methods
    
    /** Send generation request to Python brain */
    void sendGenerateRequest(const juce::String& text, float motivation, float chaos, float vulnerability);
    
    /** Send ping to check if brain server is alive */
    void sendPing();
    
    /** Get connection status */
    bool isConnected() const { return connected; }
    
    //==============================================================================
    // Testing interface (public for unit tests)
    
    /** Parse MIDI events from JSON (exposed for testing) */
    void parseMidiEventsFromJSON(const juce::String& jsonString);

private:
    //==============================================================================
    // OSC Communication
    juce::OSCSender oscSender;
    juce::OSCReceiver oscReceiver;
    bool connected;
    
    // MIDI output buffer (thread-safe)
    juce::MidiBuffer pendingMidi;
    juce::CriticalSection midiLock;
    
    // OSC message handling
    void oscMessageReceived(const juce::OSCMessage& message) override;
    
    // MIDI scheduling helpers
    void scheduleMidiEvent(int pitch, int velocity, int channel, int tick);
    int ticksPerQuarterNote;
    double currentSampleRate;
    int currentTempoBpm;  // Track tempo for accurate timing
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (DAiWBridgeAudioProcessor)
};

