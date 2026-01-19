/**
 * Intent IR v1 Usage Example
 * 
 * Demonstrates how to use IntentFrame in new code.
 * This example shows the recommended pattern for using Intent IR v1.
 */

#include "engine/KellyBrain.h"
#include "midi/MidiGenerator.h"
#include "common/IntentIRAdapter.h"
#include "shared/include/kmidi/IntentIR.h"
#include <iostream>
#include <iomanip>

using namespace kelly;

void example1_basic_usage() {
    std::cout << "\n=== Example 1: Basic IntentFrame Usage ===\n";
    
    KellyBrain brain;
    brain.initialize("./data");
    
    // Create IntentFrame from text
    IntentFrame frame = brain.fromTextToIntentFrame("I feel lost and alone");
    
    // Validate and clamp (IMPORTANT: Do this before using the frame)
    prepareIntentFrame(frame);
    
    // Generate MIDI directly from IntentFrame
    GeneratedMidi midi = brain.generateMidiFromIntentFrame(frame, 8);
    
    std::cout << "Generated MIDI:\n";
    std::cout << "  BPM: " << midi.bpm << "\n";
    std::cout << "  Bars: " << midi.bars << "\n";
    std::cout << "  Chords: " << midi.chords.size() << "\n";
    std::cout << "  Melody notes: " << midi.melody.size() << "\n";
    std::cout << "  Bass notes: " << midi.bass.size() << "\n";
}

void example2_emotion_based() {
    std::cout << "\n=== Example 2: Emotion-Based Generation ===\n";
    
    KellyBrain brain;
    brain.initialize("./data");
    
    // Create IntentFrame from emotion
    IntentFrame frame = brain.fromEmotionToIntentFrame("joy", 0.8f);
    prepareIntentFrame(frame);
    
    std::cout << "Emotion: joy (intensity: 0.8)\n";
    std::cout << "  Valence: " << std::fixed << std::setprecision(2) 
              << frame.emotion.valence << "\n";
    std::cout << "  Arousal: " << frame.emotion.arousal << "\n";
    std::cout << "  Tempo bias: " << frame.music.tempo_bias << "\n";
    std::cout << "  Mode preference: " 
              << (frame.music.mode_preference > 0 ? "major" : "minor") << "\n";
    
    GeneratedMidi midi = brain.generateMidiFromIntentFrame(frame, 4);
    std::cout << "Generated " << midi.bars << " bars at " << midi.bpm << " BPM\n";
}

void example3_midi_generator_direct() {
    std::cout << "\n=== Example 3: Using MidiGenerator Directly ===\n";
    
    KellyBrain brain;
    brain.initialize("./data");
    MidiGenerator generator;
    
    // Create IntentFrame
    IntentFrame frame = brain.fromTextToIntentFrame("I feel creative");
    prepareIntentFrame(frame);
    
    // Use MidiGenerator with IntentFrame
    GeneratedMidi midi = generator.generate(
        frame,           // IntentFrame
        8,              // bars
        0.6f,           // complexity
        0.5f,           // humanize
        0.2f,           // feel
        0.8f            // dynamics
    );
    
    std::cout << "Generated with custom parameters:\n";
    std::cout << "  Complexity: 0.6\n";
    std::cout << "  Humanize: 0.5\n";
    std::cout << "  Feel: 0.2\n";
    std::cout << "  Dynamics: 0.8\n";
    std::cout << "  Result: " << midi.bars << " bars, " << midi.bpm << " BPM\n";
}

void example4_journey() {
    std::cout << "\n=== Example 4: Emotional Journey ===\n";
    
    KellyBrain brain;
    brain.initialize("./data");
    
    // Create a journey from current to desired state
    SideA current;
    current.description = "I feel anxious";
    current.intensity = 0.7f;
    
    SideB desired;
    desired.description = "I want to feel calm";
    desired.intensity = 0.6f;
    
    IntentFrame frame = brain.fromJourneyToIntentFrame(current, desired);
    prepareIntentFrame(frame);
    
    std::cout << "Journey: " << current.description << " → " << desired.description << "\n";
    std::cout << "  Valence: " << frame.emotion.valence << "\n";
    std::cout << "  Arousal: " << frame.emotion.arousal << "\n";
    
    GeneratedMidi midi = brain.generateMidiFromIntentFrame(frame, 8);
    std::cout << "Generated journey music: " << midi.bars << " bars\n";
}

void example5_json_serialization() {
    std::cout << "\n=== Example 5: JSON Serialization ===\n";
    
    KellyBrain brain;
    brain.initialize("./data");
    
    IntentFrame frame = brain.fromTextToIntentFrame("I feel hopeful");
    prepareIntentFrame(frame);
    
    // Serialize to JSON (for logging/debugging)
    char* json = intent_frame_to_json(&frame);
    if (json) {
        std::cout << "IntentFrame JSON:\n" << json << "\n";
        free(json);
    }
    
    // Deserialize from JSON
    const char* json_str = R"({"meta":{"ir_version":1},"emotion":{"valence":0.5,"arousal":0.7}})";
    IntentFrame frame2;
    if (intent_frame_from_json(json_str, &frame2)) {
        std::cout << "Deserialized frame:\n";
        std::cout << "  Valence: " << frame2.emotion.valence << "\n";
        std::cout << "  Arousal: " << frame2.emotion.arousal << "\n";
    }
}

void example6_backward_compatibility() {
    std::cout << "\n=== Example 6: Backward Compatibility ===\n";
    
    KellyBrain brain;
    brain.initialize("./data");
    
    // Old way (still works)
    IntentResult result = brain.fromText("I feel sad");
    GeneratedMidi midi1 = brain.generateMidi(result, 4);
    
    // New way (preferred)
    IntentFrame frame = brain.fromTextToIntentFrame("I feel sad");
    prepareIntentFrame(frame);
    GeneratedMidi midi2 = brain.generateMidiFromIntentFrame(frame, 4);
    
    std::cout << "Both methods work:\n";
    std::cout << "  IntentResult path: " << midi1.bpm << " BPM\n";
    std::cout << "  IntentFrame path: " << midi2.bpm << " BPM\n";
}

void example7_conversion() {
    std::cout << "\n=== Example 7: Converting Between Formats ===\n";
    
    KellyBrain brain;
    brain.initialize("./data");
    
    // Start with IntentResult (old code)
    IntentResult result = brain.fromText("I feel energetic");
    
    // Convert to IntentFrame (new format)
    IntentFrame frame = convertIntentResultToIntentIR(result);
    prepareIntentFrame(frame);
    
    // Use IntentFrame with new methods
    GeneratedMidi midi = brain.generateMidiFromIntentFrame(frame, 8);
    
    std::cout << "Converted IntentResult → IntentFrame\n";
    std::cout << "  Generated: " << midi.bars << " bars\n";
    
    // Convert back if needed
    IntentResult result2 = convertIntentIRToIntentResult(frame);
    std::cout << "  Converted back: tempo = " << result2.tempoBpm << " BPM\n";
}

int main() {
    std::cout << "Intent IR v1 Usage Examples\n";
    std::cout << "===========================\n";
    
    try {
        example1_basic_usage();
        example2_emotion_based();
        example3_midi_generator_direct();
        example4_journey();
        example5_json_serialization();
        example6_backward_compatibility();
        example7_conversion();
        
        std::cout << "\n=== All Examples Complete ===\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
