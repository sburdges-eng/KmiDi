#include "core/intent_ir/IntentFrame.h"
#include "core/intent_ir/IntentFrameAdapter.h"
#include "engine/VADSystem.h"
#include <cassert>
#include <thread>

using namespace kelly;
using namespace kelly::intent_ir;

void test_ir_consumption() {
    // Create test IntentFrame
    IntentFrame frame;
    frame.meta.ir_version = 1;
    frame.meta.intent_id = 0x123;
    frame.meta.session_id = 0x456;
    
    frame.emotion.valence = 0.5f;
    frame.emotion.arousal = 0.6f;
    frame.emotion.dominance = 0.7f;
    frame.emotion.discrete_id = -1;
    frame.emotion.intensity = 1.0f;
    frame.emotion.confidence = 1.0f;
    
    // Test VAD extraction
    VADState vad = emotionToVAD(frame.emotion);
    assert(vad.valence == 0.5f);
    assert(vad.arousal == 0.6f);
    assert(vad.dominance == 0.7f);
}

void test_parameter_derivation() {
    IntentFrame frame;
    frame.music.tempo_bias = 0.5f;
    frame.music.rhythmic_density = 0.7f;
    frame.music.groove_strength = 0.8f;
    
    MusicalParameters params = musicToParams(frame.music);
    assert(params.tempoBpm > 120); // Should be faster with positive bias
    assert(params.textureDensity == frame.music.texture_density);
}

void test_thread_safety() {
    IntentFrameSnapshot snapshot;
    IntentFrame frame;
    frame.meta.intent_id = 0x123;
    
    // Update from one thread
    std::thread t1([&snapshot, &frame]() {
        snapshot.update(frame);
    });
    
    // Read from another thread
    std::thread t2([&snapshot]() {
        const auto& meta = snapshot.getMeta();
        (void)meta; // Suppress unused warning
    });
    
    t1.join();
    t2.join();
}

void test_snapshot_isolation() {
    IntentFrameSnapshot snapshot1;
    IntentFrameSnapshot snapshot2;
    
    IntentFrame frame1;
    frame1.meta.intent_id = 0x111;
    
    IntentFrame frame2;
    frame2.meta.intent_id = 0x222;
    
    snapshot1.update(frame1);
    snapshot2.update(frame2);
    
    assert(snapshot1.getMeta().intent_id == 0x111);
    assert(snapshot2.getMeta().intent_id == 0x222);
}

int main() {
    test_ir_consumption();
    test_parameter_derivation();
    test_thread_safety();
    test_snapshot_isolation();
    
    return 0;
}
