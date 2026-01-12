#include "plugin_editor.h"

namespace kelly {

PluginEditor::PluginEditor(PluginProcessor& p)
    : AudioProcessorEditor(&p), processor(p) {
    setSize(400, 300);
}

void PluginEditor::paint(juce::Graphics& g) {
    g.fillAll(getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId));
    
    g.setColour(juce::Colours::white);
    g.setFont(15.0f);
    g.drawFittedText("Kelly Emotion Processor", getLocalBounds(), juce::Justification::centred, 1);
}

void PluginEditor::resized() {
    // Layout components
}

} // namespace kelly
