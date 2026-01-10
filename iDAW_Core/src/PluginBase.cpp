/*
  ==============================================================================

    PluginBase.cpp
    Created: 2025
    Author: iDAW Team

    Base class implementation for all iDAW art-themed plugins

  ==============================================================================
*/

#include "PluginBase.h"

//==============================================================================
PluginBase::PluginBase(const BusesProperties& ioLayouts)
    : AudioProcessor(ioLayouts),
      memoryManager(MemoryManager::getInstance()),
      parameters(std::make_unique<juce::AudioProcessorValueTreeState>(*this, nullptr, "PARAMETERS", juce::AudioProcessorValueTreeState::ParameterLayout()))
{
    // Initialize parameters (called after parameters is created)
    initializeParameters();
}

PluginBase::~PluginBase()
{
}

//==============================================================================
void PluginBase::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    currentSampleRate = sampleRate;
    currentBlockSize = samplesPerBlock;
    
    // Ensure we're not on audio thread
    assertNotAudioThread();
}

void PluginBase::releaseResources()
{
    assertNotAudioThread();
}

#ifndef JucePlugin_PreferredChannelConfigurations
bool PluginBase::isBusesLayoutSupported (const BusesLayout& layouts) const
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

void PluginBase::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;
    
    // RT-safe processing
    processAudio(buffer, midiMessages);
}

//==============================================================================
void PluginBase::getStateInformation (juce::MemoryBlock& destData)
{
    assertNotAudioThread();
    
    if (parameters)
    {
        auto state = parameters->copyState();
        std::unique_ptr<juce::XmlElement> xml (state.createXml());
        copyXmlToBinary (*xml, destData);
    }
}

void PluginBase::setStateInformation (const void* data, int sizeInBytes)
{
    assertNotAudioThread();
    
    if (parameters)
    {
        std::unique_ptr<juce::XmlElement> xmlState (getXmlFromBinary (data, sizeInBytes));

        if (xmlState.get() != nullptr)
            if (xmlState->hasTagName (parameters->state.getType()))
                parameters->replaceState (juce::ValueTree::fromXml (*xmlState));
    }
}

//==============================================================================
bool PluginBase::isAudioThread() const noexcept
{
    return juce::MessageManager::getInstance()->isThisTheMessageThread() == false;
}

void PluginBase::assertNotAudioThread() const
{
    jassert(isAudioThread() == false);
}
