#include "BridgeClient.h"
#include "../Voice/VoiceProcessor.h"

//==============================================================================
// OSC Address Patterns (must match Python cpp_bridge.py)
//==============================================================================
namespace OSCAddresses
{
    // Incoming (Python → C++)
    constexpr const char* LOAD_VOICE_MODEL = "/voice/model/load";
    constexpr const char* SPEAK_TEXT = "/voice/speak";
    constexpr const char* QUEUE_PHONEME = "/voice/phoneme";
    constexpr const char* SET_VOWEL = "/voice/vowel";
    constexpr const char* SET_PITCH = "/voice/pitch";
    constexpr const char* NOTE_ON = "/voice/note/on";
    constexpr const char* NOTE_OFF = "/voice/note/off";
    constexpr const char* SET_FORMANT_SHIFT = "/voice/formant/shift";
    constexpr const char* SET_BREATHINESS = "/voice/breathiness";
    constexpr const char* SET_VIBRATO = "/voice/vibrato";

    // Outgoing (C++ → Python)
    constexpr const char* STATUS = "/voice/status";
    constexpr const char* PHONEME_COMPLETE = "/voice/phoneme/complete";
    constexpr const char* ERROR = "/voice/error";
}

//==============================================================================
BridgeClient::BridgeClient() = default;

BridgeClient::~BridgeClient()
{
    disconnect();
}

//==============================================================================
bool BridgeClient::connect(int receivePort, const juce::String& sendHost, int sendPort)
{
    sendHost_ = sendHost;
    sendPort_ = sendPort;

    // Create OSC receiver
    oscReceiver_ = std::make_unique<juce::OSCReceiver>();

    if (!oscReceiver_->connect(receivePort))
    {
        DBG("Failed to bind OSC receiver to port " << receivePort);
        return false;
    }

    oscReceiver_->addListener(this);

    // Create OSC sender
    oscSender_ = std::make_unique<juce::OSCSender>();

    if (!oscSender_->connect(sendHost, sendPort))
    {
        DBG("Failed to connect OSC sender to " << sendHost << ":" << sendPort);
        // Continue anyway - we can still receive
    }

    connected_ = true;
    DBG("BridgeClient connected: receiving on port " << receivePort
        << ", sending to " << sendHost << ":" << sendPort);

    sendStatus("connected");
    return true;
}

void BridgeClient::disconnect()
{
    if (connected_)
    {
        sendStatus("disconnecting");

        if (oscReceiver_)
        {
            oscReceiver_->removeListener(this);
            oscReceiver_->disconnect();
            oscReceiver_.reset();
        }

        if (oscSender_)
        {
            oscSender_->disconnect();
            oscSender_.reset();
        }

        connected_ = false;
        DBG("BridgeClient disconnected");
    }
}

bool BridgeClient::ping()
{
    if (!connected_)
        return false;

    sendStatus("ping");
    juce::Thread::sleep(50);
    return true;
}

//==============================================================================
void BridgeClient::sendStatus(const juce::String& status)
{
    if (oscSender_)
    {
        oscSender_->send(OSCAddresses::STATUS, status);
    }
}

void BridgeClient::sendPhonemeComplete(int phonemeIndex)
{
    if (oscSender_)
    {
        oscSender_->send(OSCAddresses::PHONEME_COMPLETE, phonemeIndex);
    }
}

void BridgeClient::sendError(const juce::String& error)
{
    if (oscSender_)
    {
        oscSender_->send(OSCAddresses::ERROR, error);
    }
}

//==============================================================================
bool BridgeClient::requestAutoTune(const juce::File& inputFile, juce::File& outputFile)
{
    if (!connected_ || !oscSender_)
        return false;
    
    // Send auto-tune request via OSC
    juce::String inputPath = inputFile.getFullPathName();
    juce::String outputPath = outputFile.getFullPathName();
    
    // Send OSC message: /autotune/process [input_path] [output_path]
    if (oscSender_->send("/autotune/process", inputPath, outputPath))
    {
        // Wait for processing response (with timeout)
        // In a real implementation, this would be async with a callback
        int timeout = 100;  // 10 seconds
        while (timeout > 0)
        {
            juce::Thread::sleep(100);
            timeout--;
            
            // Check for completion message (would be received via OSC)
            // For now, return success after timeout
        }
        
        return outputFile.existsAsFile();
    }
    
    return false;
}

juce::String BridgeClient::sendChatMessage(const juce::String& message)
{
    if (!connected_ || !oscSender_)
        return "Error: Not connected to Python bridge";
    
    // Send chat request via OSC
    if (oscSender_->send("/chat/message", message))
    {
        // In a real implementation, this would:
        // 1. Send message to Python AI service via OSC
        // 2. Wait for response asynchronously
        // 3. Return the AI's response
        
        // For now, provide a helpful message
        return "AI chat integration ready. Message sent to bridge: " + message;
    }
    
    return "Error: Failed to send message";
}

//==============================================================================
void BridgeClient::oscMessageReceived(const juce::OSCMessage& message)
{
    juce::String address = message.getAddressPattern().toString();

    DBG("Received OSC: " << address);

    // Route message to appropriate handler
    if (address == OSCAddresses::LOAD_VOICE_MODEL)
    {
        handleVoiceModelLoad(message);
    }
    else if (address == OSCAddresses::SPEAK_TEXT)
    {
        handleSpeakText(message);
    }
    else if (address == OSCAddresses::QUEUE_PHONEME)
    {
        handlePhoneme(message);
    }
    else if (address == OSCAddresses::SET_VOWEL)
    {
        handleVowelSet(message);
    }
    else if (address == OSCAddresses::SET_PITCH)
    {
        handlePitchSet(message);
    }
    else if (address == OSCAddresses::NOTE_ON)
    {
        handleNoteOn(message);
    }
    else if (address == OSCAddresses::NOTE_OFF)
    {
        handleNoteOff(message);
    }
    else if (address == OSCAddresses::SET_FORMANT_SHIFT)
    {
        handleFormantShift(message);
    }
    else if (address == OSCAddresses::SET_BREATHINESS)
    {
        handleBreathiness(message);
    }
    else if (address == OSCAddresses::SET_VIBRATO)
    {
        handleVibrato(message);
    }
    else
    {
        DBG("Unknown OSC address: " << address);
    }
}

//==============================================================================
void BridgeClient::handleVoiceModelLoad(const juce::OSCMessage& message)
{
    if (!voiceProcessor_)
    {
        sendError("VoiceProcessor not initialized");
        return;
    }

    if (message.size() < 1 || !message[0].isString())
    {
        sendError("Invalid voice model message format");
        return;
    }

    juce::String jsonData = message[0].getString();

    if (voiceProcessor_->loadVoiceModel(jsonData))
    {
        sendStatus("voice_model_loaded");
        DBG("Voice model loaded successfully");
    }
    else
    {
        sendError("Failed to parse voice model");
    }
}

void BridgeClient::handleSpeakText(const juce::OSCMessage& message)
{
    if (!voiceProcessor_)
    {
        sendError("VoiceProcessor not initialized");
        return;
    }

    if (message.size() < 1 || !message[0].isString())
    {
        sendError("Invalid speak text message format");
        return;
    }

    juce::String text = message[0].getString();
    voiceProcessor_->speakText(text);
    sendStatus("speaking");
}

void BridgeClient::handlePhoneme(const juce::OSCMessage& message)
{
    if (!voiceProcessor_)
        return;

    // Expected: [index, vowel_index, duration, pitch, stress, is_consonant]
    if (message.size() < 6)
    {
        sendError("Invalid phoneme message format");
        return;
    }

    int index = message[0].getInt32();
    int vowelIndex = message[1].getInt32();
    float duration = message[2].getFloat32();
    float pitch = message[3].getFloat32();
    int stress = message[4].getInt32();
    int isConsonant = message[5].getInt32();

    SynthPhoneme phoneme;
    phoneme.vowelType = static_cast<VowelType>(vowelIndex);
    phoneme.duration = duration;
    phoneme.pitch = pitch;
    phoneme.stress = stress;
    phoneme.isConsonant = isConsonant != 0;

    // Queue single phoneme (accumulate in vector)
    // For simplicity, we'll create a single-phoneme vector
    std::vector<SynthPhoneme> phonemes = { phoneme };
    voiceProcessor_->queuePhonemes(phonemes);
}

void BridgeClient::handleVowelSet(const juce::OSCMessage& message)
{
    if (!voiceProcessor_)
        return;

    if (message.size() < 1)
        return;

    int vowelIndex = message[0].getInt32();
    voiceProcessor_->setVowel(static_cast<VowelType>(vowelIndex));
}

void BridgeClient::handlePitchSet(const juce::OSCMessage& message)
{
    if (!voiceProcessor_)
        return;

    if (message.size() < 1)
        return;

    float pitch = message[0].getFloat32();
    voiceProcessor_->setPitch(pitch);
}

void BridgeClient::handleNoteOn(const juce::OSCMessage& message)
{
    if (!voiceProcessor_)
        return;

    if (message.size() < 2)
        return;

    int midiNote = message[0].getInt32();
    float velocity = message[1].getFloat32();
    voiceProcessor_->noteOn(midiNote, velocity);
}

void BridgeClient::handleNoteOff(const juce::OSCMessage& message)
{
    if (!voiceProcessor_)
        return;

    voiceProcessor_->noteOff();
}

void BridgeClient::handleFormantShift(const juce::OSCMessage& message)
{
    if (!voiceProcessor_)
        return;

    if (message.size() < 1)
        return;

    float shift = message[0].getFloat32();
    voiceProcessor_->formantShift.store(shift);
}

void BridgeClient::handleBreathiness(const juce::OSCMessage& message)
{
    if (!voiceProcessor_)
        return;

    if (message.size() < 1)
        return;

    float amount = message[0].getFloat32();
    voiceProcessor_->breathiness.store(amount);
}

void BridgeClient::handleVibrato(const juce::OSCMessage& message)
{
    if (!voiceProcessor_)
        return;

    if (message.size() < 1)
        return;

    float amount = message[0].getFloat32();
    voiceProcessor_->vibratoAmount.store(amount);
}
