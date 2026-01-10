/*
  ==============================================================================

    PluginBase.h
    Created: 2025
    Author: iDAW Team

    Base class for all iDAW art-themed plugins

  ==============================================================================
*/

#pragma once

#include <JuceHeader.h>
#include "MemoryManager.h"
#include "SafetyUtils.h"

//==============================================================================
/**
    Base class for all iDAW plugins
    
    Provides common functionality:
    - Dual-heap memory management (Side A/B)
    - RT-safe audio processing
    - Parameter management
    - State persistence
*/
class PluginBase : public juce::AudioProcessor
{
public:
    //==============================================================================
    PluginBase(const BusesProperties& ioLayouts = BusesProperties());
    ~PluginBase() override;

    //==============================================================================
    // AudioProcessor overrides
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    
    #ifndef JucePlugin_PreferredChannelConfigurations
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
    #endif

    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    //==============================================================================
    // State management
    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    //==============================================================================
    // Plugin-specific overrides (implement in derived classes)
    
    /** Process audio - called from processBlock (RT-safe) */
    virtual void processAudio(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) noexcept = 0;
    
    /** Initialize plugin-specific parameters */
    virtual void initializeParameters() {}
    
    /** Get plugin name */
    virtual juce::String getPluginName() const = 0;
    
    /** Get plugin description */
    virtual juce::String getPluginDescription() const = 0;

protected:
    //==============================================================================
    // Memory management
    MemoryManager* memoryManager;
    
    // Audio state
    double currentSampleRate = 44100.0;
    int currentBlockSize = 512;
    
    // Parameter management
    std::unique_ptr<juce::AudioProcessorValueTreeState> parameters;
    
    // RT-safety helpers
    bool isAudioThread() const noexcept;
    void assertNotAudioThread() const;

private:
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (PluginBase)
};
