cmake_minimum_required(VERSION 3.22)
project(KellyMidiCompanion VERSION 2.0.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# JUCE - Fetch from GitHub
include(FetchContent)
FetchContent_Declare(
    JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG 8.0.4
)
FetchContent_MakeAvailable(JUCE)

# Plugin target
juce_add_plugin(KellyMidiCompanion
    COMPANY_NAME "Kelly Project"
    PLUGIN_MANUFACTURER_CODE Klly
    PLUGIN_CODE Kmcp
    
    # Plugin formats
    FORMATS AU VST3 Standalone
    
    # This is a MIDI effect, not an audio processor
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT TRUE
    NEEDS_MIDI_OUTPUT TRUE
    IS_MIDI_EFFECT TRUE
    
    PRODUCT_NAME "Kelly MIDI Companion"
    PLUGIN_DESCRIPTION "Therapeutic MIDI generation through emotion mapping. Transforms emotional states (valence, arousal, intensity) into musical MIDI patterns using a 216-node emotion thesaurus and intentional rule-breaking system."
    
    # Bundle IDs
    BUNDLE_ID "com.kellyproject.midicompanion"
    AU_MAIN_TYPE "kAudioUnitType_MIDIProcessor"
)

# Source files
target_sources(KellyMidiCompanion
    PRIVATE
        # Plugin core
        src/plugin/PluginProcessor.cpp
        src/plugin/PluginEditor.cpp
        src/plugin/PluginState.cpp
        
        # Emotion engine
        src/engine/EmotionThesaurus.cpp
        src/engine/EmotionThesaurusLoader.cpp
        src/engine/WoundProcessor.cpp
        src/engine/RuleBreakEngine.cpp
        src/engine/IntentPipeline.cpp
        
        # MIDI generation
        src/midi/ChordGenerator.cpp
        src/midi/GrooveEngine.cpp
        src/midi/MidiBuilder.cpp
        src/midi/MidiGenerator.cpp
        
        # UI
        src/ui/CassetteView.cpp
        src/ui/SidePanel.cpp
        src/ui/GenerateButton.cpp
        src/ui/KellyLookAndFeel.cpp
        src/ui/EmotionWheel.cpp
        
        # Voice synthesis (v2.0)
        src/voice/VoiceSynthesizer.cpp
        
        # Biometric input (v2.0)
        src/biometric/BiometricInput.cpp
)

# Include directories
target_include_directories(KellyMidiCompanion
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# JUCE modules
target_compile_definitions(KellyMidiCompanion
    PUBLIC
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0
        JUCE_DISPLAY_SPLASH_SCREEN=0
)

target_link_libraries(KellyMidiCompanion
    PRIVATE
        juce::juce_audio_basics
        juce::juce_audio_devices
        juce::juce_audio_formats
        juce::juce_audio_plugin_client
        juce::juce_audio_processors
        juce::juce_audio_utils
        juce::juce_core
        juce::juce_data_structures
        juce::juce_events
        juce::juce_graphics
        juce::juce_gui_basics
        juce::juce_gui_extra
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)

# Note: Code signing is handled by build_and_install.sh script
# We don't sign during build to avoid Gatekeeper issues
# The install script properly removes quarantine and signs after installation
