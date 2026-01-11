#!/usr/bin/env python3
"""
Plugin Generator Script
Generates all 11 iDAW art-themed plugins with proper structure
"""

import os
import json
from pathlib import Path

# Plugin definitions
PLUGINS = [
    {
        "name": "Pencil",
        "id": "Pencil",
        "description": "Sketching/drafting audio ideas",
        "shader": "Graphite",
        "priority": "HIGH",
        "type": "effect",
        "params": [
            ("sketchiness", 0.0, 1.0, 0.5, "Sketchiness"),
            ("detail", 0.0, 1.0, 0.7, "Detail Level"),
            ("pressure", 0.0, 1.0, 0.5, "Pressure Sensitivity")
        ]
    },
    {
        "name": "Eraser",
        "id": "Eraser",
        "description": "Audio removal/cleanup",
        "shader": "ChalkDust",
        "priority": "HIGH",
        "type": "effect",
        "params": [
            ("intensity", 0.0, 1.0, 0.5, "Erasure Intensity"),
            ("softness", 0.0, 1.0, 0.3, "Edge Softness"),
            ("threshold", 0.0, 1.0, 0.4, "Detection Threshold")
        ]
    },
    {
        "name": "Press",
        "id": "Press",
        "description": "Dynamics/compression",
        "shader": "Heartbeat",
        "priority": "HIGH",
        "type": "effect",
        "params": [
            ("ratio", 1.0, 20.0, 4.0, "Compression Ratio"),
            ("threshold", -60.0, 0.0, -12.0, "Threshold (dB)"),
            ("attack", 0.1, 100.0, 10.0, "Attack (ms)"),
            ("release", 1.0, 1000.0, 100.0, "Release (ms)")
        ]
    },
    {
        "name": "Palette",
        "id": "Palette",
        "description": "Tonal coloring/mixing",
        "shader": "Watercolor",
        "priority": "MID",
        "type": "effect",
        "params": [
            ("hue", 0.0, 360.0, 0.0, "Hue Shift"),
            ("saturation", 0.0, 2.0, 1.0, "Saturation"),
            ("brightness", 0.0, 2.0, 1.0, "Brightness"),
            ("blend", 0.0, 1.0, 0.5, "Blend Amount")
        ]
    },
    {
        "name": "Smudge",
        "id": "Smudge",
        "description": "Audio blending/smoothing",
        "shader": "Scrapbook",
        "priority": "MID",
        "type": "effect",
        "params": [
            ("amount", 0.0, 1.0, 0.5, "Smudge Amount"),
            ("radius", 0.0, 1.0, 0.3, "Blend Radius"),
            ("direction", 0.0, 1.0, 0.5, "Direction")
        ]
    },
    {
        "name": "Trace",
        "id": "Trace",
        "description": "Pattern following/automation",
        "shader": "Spirograph",
        "priority": "LOW",
        "type": "effect",
        "params": [
            ("speed", 0.0, 10.0, 1.0, "Pattern Speed"),
            ("complexity", 0.0, 1.0, 0.5, "Pattern Complexity"),
            ("feedback", 0.0, 1.0, 0.3, "Feedback Amount")
        ]
    },
    {
        "name": "Parrot",
        "id": "Parrot",
        "description": "Sample playback/mimicry",
        "shader": "Feather",
        "priority": "LOW",
        "type": "synth",
        "params": [
            ("pitch", 0.25, 4.0, 1.0, "Pitch Shift"),
            ("speed", 0.25, 4.0, 1.0, "Playback Speed"),
            ("loop", 0.0, 1.0, 0.0, "Loop Amount")
        ]
    },
    {
        "name": "Stencil",
        "id": "Stencil",
        "description": "Sidechain/ducking effect",
        "shader": "Cutout",
        "priority": "LOW",
        "type": "effect",
        "params": [
            ("depth", 0.0, 1.0, 0.5, "Ducking Depth"),
            ("attack", 0.1, 100.0, 10.0, "Attack (ms)"),
            ("release", 1.0, 1000.0, 100.0, "Release (ms)")
        ]
    },
    {
        "name": "Chalk",
        "id": "Chalk",
        "description": "Lo-fi/bitcrusher effect",
        "shader": "Dusty",
        "priority": "LOW",
        "type": "effect",
        "params": [
            ("bitdepth", 1.0, 32.0, 16.0, "Bit Depth"),
            ("downsample", 1.0, 32.0, 1.0, "Downsample Factor"),
            ("noise", 0.0, 1.0, 0.1, "Noise Amount")
        ]
    },
    {
        "name": "Brush",
        "id": "Brush",
        "description": "Modulated filter effect",
        "shader": "Brushstroke",
        "priority": "LOW",
        "type": "effect",
        "params": [
            ("cutoff", 20.0, 20000.0, 1000.0, "Cutoff (Hz)"),
            ("resonance", 0.1, 10.0, 1.0, "Resonance"),
            ("modulation", 0.0, 1.0, 0.5, "Modulation Amount")
        ]
    },
    {
        "name": "Stamp",
        "id": "Stamp",
        "description": "Stutter/repeater effect",
        "shader": "RubberStamp",
        "priority": "LOW",
        "type": "effect",
        "params": [
            ("rate", 0.1, 32.0, 1.0, "Repeat Rate"),
            ("length", 0.01, 1.0, 0.1, "Repeat Length"),
            ("feedback", 0.0, 0.9, 0.0, "Feedback")
        ]
    }
]

def generate_plugin_files(plugin, base_path):
    """Generate all files for a plugin"""
    plugin_path = base_path / plugin["id"]
    plugin_path.mkdir(exist_ok=True)
    
    # Generate header
    header_content = generate_header(plugin)
    (plugin_path / f"{plugin['id']}Processor.h").write_text(header_content)
    
    # Generate implementation
    impl_content = generate_implementation(plugin)
    (plugin_path / f"{plugin['id']}Processor.cpp").write_text(impl_content)
    
    # Generate editor header
    editor_header = generate_editor_header(plugin)
    (plugin_path / f"{plugin['id']}Editor.h").write_text(editor_header)
    
    # Generate editor implementation
    editor_impl = generate_editor_implementation(plugin)
    (plugin_path / f"{plugin['id']}Editor.cpp").write_text(editor_impl)
    
    print(f"✓ Generated {plugin['name']} plugin")

def generate_header(plugin):
    """Generate processor header file"""
    params_decl = ""
    for param_id, min_val, max_val, default_val, name in plugin["params"]:
        params_decl += f'    juce::AudioParameterFloat* {param_id}Param = nullptr;\n'
    
    return f'''/*
  ==============================================================================

    {plugin["id"]}Processor.h
    Created: 2025
    Author: iDAW Team

    {plugin["name"]} - {plugin["description"]}
    Shader: {plugin["shader"]}

  ==============================================================================
*/

#pragma once

#include "../PluginBase.h"

//==============================================================================
/**
    {plugin["name"]} Audio Processor
    
    {plugin["description"]}
*/
class {plugin["id"]}AudioProcessor : public PluginBase
{{
public:
    //==============================================================================
    {plugin["id"]}AudioProcessor();
    ~{plugin["id"]}AudioProcessor() override;

    //==============================================================================
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

    void processAudio(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) noexcept override;

    //==============================================================================
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override {{ return true; }}

    //==============================================================================
    const juce::String getName() const override {{ return "{plugin["name"]}"; }}
    juce::String getPluginName() const override {{ return "{plugin["name"]}"; }}
    juce::String getPluginDescription() const override {{ return "{plugin["description"]}"; }}

    //==============================================================================
    void initializeParameters() override;

private:
    //==============================================================================
    // Parameters
{params_decl}
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR ({plugin["id"]}AudioProcessor)
}};
'''

def generate_implementation(plugin):
    """Generate processor implementation file"""
    param_init = ""
    param_getters = ""
    param_layouts = ""
    
    for param_id, min_val, max_val, default_val, name in plugin["params"]:
        param_init += f'    {param_id}Param = parameters.getRawParameterValue("{param_id}");\n'
        param_getters += f'    auto {param_id} = {param_id}Param->load();\n'
        param_layouts += f'    layout.add(std::make_unique<juce::AudioParameterFloat>(\n'
        param_layouts += f'        "{param_id}", "{name}",\n'
        param_layouts += f'        juce::NormalisableRange<float>({min_val}f, {max_val}f),\n'
        param_layouts += f'        {default_val}f));\n'
    
    return f'''/*
  ==============================================================================

    {plugin["id"]}Processor.cpp
    Created: 2025
    Author: iDAW Team

    {plugin["name"]} - {plugin["description"]}

  ==============================================================================
*/

#include "{plugin["id"]}Processor.h"
#include "{plugin["id"]}Editor.h"

//==============================================================================
{plugin["id"]}AudioProcessor::{plugin["id"]}AudioProcessor()
    : PluginBase(BusesProperties()
        #if ! JucePlugin_IsMidiEffect
         #if ! JucePlugin_IsSynth
          .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
         #endif
          .withOutput ("Output", juce::AudioChannelSet::stereo(), true)
        #endif
          )
{{
    initializeParameters();
}}

{plugin["id"]}AudioProcessor::~{plugin["id"]}AudioProcessor()
{{
}}

//==============================================================================
void {plugin["id"]}AudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{{
    PluginBase::prepareToPlay(sampleRate, samplesPerBlock);
}}

void {plugin["id"]}AudioProcessor::releaseResources()
{{
    PluginBase::releaseResources();
}}

//==============================================================================
void {plugin["id"]}AudioProcessor::processAudio(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) noexcept
{{
    // RT-safe processing
    const int numChannels = buffer.getNumChannels();
    const int numSamples = buffer.getNumSamples();
    
    // Get parameter values
{param_getters}
    
    // Process audio
    for (int channel = 0; channel < numChannels; ++channel)
    {{
        auto* channelData = buffer.getWritePointer(channel);
        
        for (int sample = 0; sample < numSamples; ++sample)
        {{
            // TODO: Implement {plugin["name"]} processing
            // channelData[sample] = processSample(channelData[sample], ...);
        }}
    }}
}}

//==============================================================================
void {plugin["id"]}AudioProcessor::initializeParameters()
{{
    juce::AudioProcessorValueTreeState::ParameterLayout layout;
    
{param_layouts}
    parameters.replaceState(juce::ValueTree::fromXml(*juce::XmlElement("PARAMETERS")));
    
    // Get parameter pointers
{param_init}
}}

//==============================================================================
juce::AudioProcessorEditor* {plugin["id"]}AudioProcessor::createEditor()
{{
    return new {plugin["id"]}AudioProcessorEditor (*this);
}}

//==============================================================================
// This creates new instances of the plugin
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{{
    return new {plugin["id"]}AudioProcessor();
}}
'''

def generate_editor_header(plugin):
    """Generate editor header file"""
    ui_components = ""
    for param_id, _, _, _, _ in plugin["params"]:
        ui_components += f'    juce::Slider {param_id}Slider;\n'
        ui_components += f'    juce::Label {param_id}Label;\n'
        ui_components += f'    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> {param_id}Attachment;\n'
    
    return f'''/*
  ==============================================================================

    {plugin["id"]}Editor.h
    Created: 2025
    Author: iDAW Team

    {plugin["name"]} Plugin Editor

  ==============================================================================
*/

#pragma once

#include <JuceHeader.h>
#include "{plugin["id"]}Processor.h"

//==============================================================================
/**
    {plugin["name"]} Audio Processor Editor
*/
class {plugin["id"]}AudioProcessorEditor  : public juce::AudioProcessorEditor
{{
public:
    {plugin["id"]}AudioProcessorEditor ({plugin["id"]}AudioProcessor&);
    ~{plugin["id"]}AudioProcessorEditor() override;

    //==============================================================================
    void paint (juce::Graphics&) override;
    void resized() override;

private:
    //==============================================================================
    {plugin["id"]}AudioProcessor& audioProcessor;
    
    // UI Components
{ui_components}
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR ({plugin["id"]}AudioProcessorEditor)
}};
'''

def generate_editor_implementation(plugin):
    """Generate editor implementation file"""
    slider_attachments = ""
    slider_setup = ""
    slider_bounds = ""
    
    for i, (param_id, _, _, _, name) in enumerate(plugin["params"]):
        slider_attachments += f'    {param_id}Attachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(\n'
        slider_attachments += f'        audioProcessor.parameters, "{param_id}", {param_id}Slider);\n'
        
        slider_setup += f'    {param_id}Slider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);\n'
        slider_setup += f'    {param_id}Slider.setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);\n'
        slider_setup += f'    {param_id}Slider.setPopupDisplayEnabled(true, false, this);\n'
        slider_setup += f'    addAndMakeVisible(&{param_id}Slider);\n'
        slider_setup += f'\n'
        slider_setup += f'    {param_id}Label.setText("{name}", juce::dontSendNotification);\n'
        slider_setup += f'    {param_id}Label.attachToComponent(&{param_id}Slider, false);\n'
        slider_setup += f'    addAndMakeVisible(&{param_id}Label);\n'
        slider_setup += f'\n'
        
        slider_bounds += f'    {param_id}Slider.setBounds(10 + {i} * 80, 60, 70, 70);\n'
    
    return f'''/*
  ==============================================================================

    {plugin["id"]}Editor.cpp
    Created: 2025
    Author: iDAW Team

    {plugin["name"]} Plugin Editor

  ==============================================================================
*/

#include "{plugin["id"]}Editor.h"

//==============================================================================
{plugin["id"]}AudioProcessorEditor::{plugin["id"]}AudioProcessorEditor ({plugin["id"]}AudioProcessor& p)
    : AudioProcessorEditor (&p), audioProcessor (p)
{{
    setSize (400, 300);
    
    // Setup sliders
{slider_setup}
    
    // Attach parameters
{slider_attachments}
}}

{plugin["id"]}AudioProcessorEditor::~{plugin["id"]}AudioProcessorEditor()
{{
}}

//==============================================================================
void {plugin["id"]}AudioProcessorEditor::paint (juce::Graphics& g)
{{
    g.fillAll (getLookAndFeel().findColour (juce::ResizableWindow::backgroundColourId));

    g.setColour (juce::Colours::white);
    g.setFont (15.0f);
    g.drawFittedText ("{plugin["name"]}", getLocalBounds(), juce::Justification::centredTop, 1);
}}

void {plugin["id"]}AudioProcessorEditor::resized()
{{
{slider_bounds}
}}
'''

def main():
    """Main function"""
    base_path = Path(__file__).parent.parent / "iDAW_Core" / "plugins"
    base_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {len(PLUGINS)} plugins in {base_path}...\n")
    
    for plugin in PLUGINS:
        generate_plugin_files(plugin, base_path)
    
    print(f"\n✓ Generated all {len(PLUGINS)} plugins!")
    print(f"  Location: {base_path}")

if __name__ == "__main__":
    main()
