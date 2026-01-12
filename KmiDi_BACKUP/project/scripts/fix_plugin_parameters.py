#!/usr/bin/env python3
"""
Fix generated plugins to use correct parameter syntax (parameters-> instead of parameters.)
"""

import os
from pathlib import Path

def fix_plugin_file(file_path):
    """Fix parameter references in a plugin file"""
    content = file_path.read_text()
    
    # Replace parameters. with parameters->
    content = content.replace('parameters.getRawParameterValue', 'parameters->getRawParameterValue')
    content = content.replace('parameters.replaceState', 'parameters->replaceState')
    content = content.replace('parameters.copyState', 'parameters->copyState')
    content = content.replace('parameters.state', 'parameters->state')
    content = content.replace('audioProcessor.parameters,', 'audioProcessor.parameters->')
    
    # Fix initializeParameters to use parameters->createAndAddParameter
    if 'initializeParameters()' in content and 'ParameterLayout layout;' in content:
        # Replace the layout creation pattern
        content = content.replace(
            'juce::AudioProcessorValueTreeState::ParameterLayout layout;',
            'auto layout = std::make_unique<juce::AudioProcessorValueTreeState::ParameterLayout>();'
        )
        # Fix the replaceState call
        if 'parameters.replaceState' in content:
            content = content.replace(
                'parameters.replaceState(juce::ValueTree::fromXml(*juce::XmlElement("PARAMETERS")));',
                'parameters->replaceState(juce::ValueTree::fromXml(*juce::XmlElement("PARAMETERS")));'
            )
        # Fix parameter layout usage
        content = content.replace('layout.add(', 'layout->add(')
        # After layout is built, we need to create the ValueTreeState
        if 'layout->add(' in content and 'parameters->replaceState' not in content.split('layout->add(')[-1].split('parameters')[0]:
            # Insert parameter state creation before replaceState
            old_pattern = '    parameters->replaceState'
            new_pattern = '''    parameters = std::make_unique<juce::AudioProcessorValueTreeState>(
        *this, nullptr, "PARAMETERS", std::move(*layout));
    
    // Get parameter pointers
    parameters->replaceState'''
            if old_pattern in content:
                content = content.replace(old_pattern, new_pattern)
    
    file_path.write_text(content)
    print(f"✓ Fixed {file_path.name}")

def main():
    """Main function"""
    plugins_dir = Path(__file__).parent.parent / "iDAW_Core" / "plugins"
    
    if not plugins_dir.exists():
        print(f"Error: Plugins directory not found: {plugins_dir}")
        return
    
    print(f"Fixing parameter syntax in plugins...\n")
    
    for plugin_dir in plugins_dir.iterdir():
        if plugin_dir.is_dir():
            processor_cpp = plugin_dir / f"{plugin_dir.name}Processor.cpp"
            editor_cpp = plugin_dir / f"{plugin_dir.name}Editor.cpp"
            
            if processor_cpp.exists():
                fix_plugin_file(processor_cpp)
            if editor_cpp.exists():
                fix_plugin_file(editor_cpp)
    
    print(f"\n✓ Fixed all plugin parameter references!")

if __name__ == "__main__":
    main()
