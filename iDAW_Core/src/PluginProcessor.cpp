/*
  ==============================================================================

    PluginProcessor.cpp
    Created: 2025
    Author: DAiW Team

    DAiW Bridge Plugin - Connects DAW to Python Brain via OSC

  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
DAiWBridgeAudioProcessor::DAiWBridgeAudioProcessor()
#ifndef JucePlugin_PreferredChannelConfigurations
     : AudioProcessor (BusesProperties()
                     #if ! JucePlugin_IsMidiEffect
                      #if ! JucePlugin_IsSynth
                       .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
                      #endif
                       .withOutput ("Output", juce::AudioChannelSet::stereo(), true)
                     #endif
                       ),
#endif
    connected(false),
    ticksPerQuarterNote(480),
    currentSampleRate(44100.0),
    currentTempoBpm(120)
{
    // Initialize OSC sender (sends to Python brain server on port 9000)
    if (oscSender.connect("127.0.0.1", 9000))
    {
        DBG("OSC sender connected to brain server (127.0.0.1:9000)");
    }
    else
    {
        DBG("Failed to connect OSC sender to brain server");
        connected = false;
    }
    
    // Initialize OSC receiver (receives from Python brain server on port 9001)
    if (oscReceiver.connect(9001))
    {
        oscReceiver.addListener(this);
        DBG("OSC receiver listening on port 9001");
        // Connection status will be confirmed by pong response
        connected = false;  // Start as false, will be set true on pong
    }
    else
    {
        DBG("Failed to connect OSC receiver on port 9001");
        connected = false;
    }
}

DAiWBridgeAudioProcessor::~DAiWBridgeAudioProcessor()
{
    oscReceiver.removeListener(this);
}

//==============================================================================
const juce::String DAiWBridgeAudioProcessor::getName() const
{
    return JucePlugin_Name;
}

bool DAiWBridgeAudioProcessor::acceptsMidi() const
{
   #if JucePlugin_WantsMidiInput
    return true;
   #else
    return false;
   #endif
}

bool DAiWBridgeAudioProcessor::producesMidi() const
{
   #if JucePlugin_ProducesMidiOutput
    return true;
   #else
    return false;
   #endif
}

bool DAiWBridgeAudioProcessor::isMidiEffect() const
{
   #if JucePlugin_IsMidiEffect
    return true;
   #else
    return false;
   #endif
}

double DAiWBridgeAudioProcessor::getTailLengthSeconds() const
{
    return 0.0;
}

int DAiWBridgeAudioProcessor::getNumPrograms()
{
    return 1;
}

int DAiWBridgeAudioProcessor::getCurrentProgram()
{
    return 0;
}

void DAiWBridgeAudioProcessor::setCurrentProgram (int index)
{
}

const juce::String DAiWBridgeAudioProcessor::getProgramName (int index)
{
    return {};
}

void DAiWBridgeAudioProcessor::changeProgramName (int index, const juce::String& newName)
{
}

//==============================================================================
void DAiWBridgeAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    currentSampleRate = sampleRate;
}

void DAiWBridgeAudioProcessor::releaseResources()
{
}

#ifndef JucePlugin_PreferredChannelConfigurations
bool DAiWBridgeAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
  #if JucePlugin_IsMidiEffect
    juce::ignoreUnused (layouts);
    return true;
  #else
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()
     && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

    #if ! JucePlugin_IsSynth
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;
    #endif

    return true;
  #endif
}
#endif

void DAiWBridgeAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;
    auto totalNumInputChannels  = getTotalNumInputChannels();
    auto totalNumOutputChannels = getTotalNumOutputChannels();

    // Clear output buffer
    for (auto i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear (i, 0, buffer.getNumSamples());

    // Process audio (passthrough for now)
    for (auto channel = 0; channel < totalNumInputChannels; ++channel)
    {
        auto* channelData = buffer.getWritePointer (channel);
        // Audio passthrough - no processing yet
    }

    // Add pending MIDI events to output
    {
        juce::ScopedLock lock(midiLock);
        midiMessages.addEvents(pendingMidi, 0, buffer.getNumSamples(), 0);
        pendingMidi.clear();
    }
}

//==============================================================================
bool DAiWBridgeAudioProcessor::hasEditor() const
{
    return true;
}

juce::AudioProcessorEditor* DAiWBridgeAudioProcessor::createEditor()
{
    return new DAiWBridgeAudioProcessorEditor (*this);
}

//==============================================================================
void DAiWBridgeAudioProcessor::getStateInformation (juce::MemoryBlock& destData)
{
}

void DAiWBridgeAudioProcessor::setStateInformation (const void* data, int sizeInBytes)
{
}

//==============================================================================
// DAiW-specific methods

void DAiWBridgeAudioProcessor::sendGenerateRequest(const juce::String& text, float motivation, float chaos, float vulnerability)
{
    if (!connected)
    {
        DBG("Not connected to brain server");
        return;
    }
    
    juce::OSCMessage msg("/daiw/generate");
    msg.addString(text);
    msg.addFloat32(motivation);
    msg.addFloat32(chaos);
    msg.addFloat32(vulnerability);
    
    if (!oscSender.send(msg))
    {
        DBG("Failed to send generate request");
    }
}

void DAiWBridgeAudioProcessor::sendPing()
{
    // Always try to send ping, even if not connected
    // This helps establish/re-establish connection
    juce::OSCMessage msg("/daiw/ping");
    if (!oscSender.send(msg))
    {
        DBG("Failed to send ping");
        connected = false;
    }
}

void DAiWBridgeAudioProcessor::oscMessageReceived(const juce::OSCMessage& message)
{
    juce::String address = message.getAddressPattern().toString();
    
    if (address == "/daiw/result")
    {
        if (message.size() > 0 && message[0].isString())
        {
            juce::String jsonString = message[0].getString();
            DBG("Received result from brain server");
            parseMidiEventsFromJSON(jsonString);
        }
        else
        {
            DBG("Received /daiw/result but no JSON data");
        }
    }
    else if (address == "/daiw/pong")
    {
        DBG("Received pong from brain server - connection confirmed");
        connected = true;
    }
    else if (address == "/daiw/error")
    {
        if (message.size() > 0 && message[0].isString())
        {
            juce::String errorJson = message[0].getString();
            DBG("Error from brain server: " + errorJson);
            
            // Try to parse error message
            auto errorObj = juce::JSON::parse(errorJson);
            if (errorObj.hasProperty("message"))
            {
                juce::String errorMsg = errorObj.getProperty("message").toString();
                DBG("Error message: " + errorMsg);
            }
        }
        connected = false;  // Mark as disconnected on error
    }
    else
    {
        DBG("Received unknown OSC message: " + address);
    }
}

void DAiWBridgeAudioProcessor::parseMidiEventsFromJSON(const juce::String& jsonString)
{
    // Parse JSON and extract MIDI events
    auto json = juce::JSON::parse(jsonString);
    
    if (json.isVoid() || !json.hasProperty("midi_events"))
    {
        DBG("Invalid JSON or missing midi_events property");
        return;
    }
    
    // Get tempo and PPQ for accurate timing
    int tempoBpm = currentTempoBpm;  // Default to current tempo
    int ppq = ticksPerQuarterNote;   // Default to current PPQ
    
    if (json.hasProperty("plan"))
    {
        auto plan = json.getProperty("plan");
        if (plan.hasProperty("tempo_bpm"))
        {
            tempoBpm = (int)plan.getProperty("tempo_bpm");
            currentTempoBpm = tempoBpm;  // Update stored tempo
        }
    }
    if (json.hasProperty("ppq"))
    {
        ppq = (int)json.getProperty("ppq");
        ticksPerQuarterNote = ppq;  // Update stored PPQ
    }
    
    // Calculate samples per tick (accounting for tempo)
    double secondsPerTick = (60.0 / (tempoBpm * ppq));
    double samplesPerTick = secondsPerTick * currentSampleRate;
    
    auto midiEventsArray = json.getProperty("midi_events");
    
    if (!midiEventsArray.isArray())
    {
        DBG("midi_events is not an array");
        return;
    }
    
    juce::ScopedLock lock(midiLock);
    
    // Clear existing events
    pendingMidi.clear();
    
    for (int i = 0; i < midiEventsArray.size(); ++i)
    {
        auto event = midiEventsArray[i];
        
        if (!event.hasProperty("type") || !event.hasProperty("pitch"))
            continue;
        
        juce::String type = event.getProperty("type").toString();
        int pitch = (int)event.getProperty("pitch");
        
        // Validate pitch range
        if (pitch < 0 || pitch > 127)
            continue;
        
        int velocity = event.hasProperty("velocity") ? (int)event.getProperty("velocity") : 80;
        velocity = juce::jlimit(0, 127, velocity);  // Clamp velocity
        
        int channel = event.hasProperty("channel") ? (int)event.getProperty("channel") : 1;
        channel = juce::jlimit(1, 16, channel);  // Clamp channel (1-16, not 0-15 for MIDI)
        
        int tick = event.hasProperty("tick") ? (int)event.getProperty("tick") : 0;
        
        // Convert tick to sample position (with tempo consideration)
        int sampleOffset = (int)(tick * samplesPerTick);
        sampleOffset = juce::jmax(0, sampleOffset);  // Ensure non-negative
        
        if (type == "note_on")
        {
            pendingMidi.addEvent(
                juce::MidiMessage::noteOn(channel, pitch, (juce::uint8)velocity),
                sampleOffset
            );
        }
        else if (type == "note_off")
        {
            pendingMidi.addEvent(
                juce::MidiMessage::noteOff(channel, pitch),
                sampleOffset
            );
        }
    }
    
    DBG("Parsed " + juce::String(pendingMidi.getNumEvents()) + " MIDI events from JSON");
}

void DAiWBridgeAudioProcessor::scheduleMidiEvent(int pitch, int velocity, int channel, int tick)
{
    juce::ScopedLock lock(midiLock);
    
    int sampleOffset = (int)((tick / (double)ticksPerQuarterNote) * currentSampleRate);
    
    pendingMidi.addEvent(
        juce::MidiMessage::noteOn(channel, pitch, (juce::uint8)velocity),
        sampleOffset
    );
}

//==============================================================================
// This creates new instances of the plugin..
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new DAiWBridgeAudioProcessor();
}

