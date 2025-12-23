// penta_core_wrapper.cpp
// Penta Core wrapper for Android, integrating with Oboe audio system

#include <jni.h>
#include <android/log.h>
#include <memory>
#include <mutex>
#include <vector>
#include <atomic>
#include <algorithm>
#include <cstring>

#include "penta/harmony/HarmonyEngine.h"
#include "penta/groove/GrooveEngine.h"
#include "penta/diagnostics/DiagnosticsEngine.h"
#include "penta/common/RTTypes.h"

#define LOG_TAG "PentaCore"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

using namespace penta;
using namespace penta::harmony;
using namespace penta::groove;
using namespace penta::diagnostics;

// Global Penta Core instance
struct PentaCoreInstance {
    std::unique_ptr<HarmonyEngine> harmonyEngine;
    std::unique_ptr<GrooveEngine> grooveEngine;
    std::unique_ptr<DiagnosticsEngine> diagnosticsEngine;
    
    double sampleRate;
    std::atomic<bool> initialized{false};
    std::mutex mutex;
    
    // MIDI note buffer (thread-safe)
    std::vector<Note> activeNotes;
    std::mutex notesMutex;
    
    PentaCoreInstance(double sr) : sampleRate(sr) {
        HarmonyEngine::Config harmonyConfig;
        harmonyConfig.sampleRate = sr;
        harmonyEngine = std::make_unique<HarmonyEngine>(harmonyConfig);
        
        GrooveEngine::Config grooveConfig;
        grooveConfig.sampleRate = sr;
        grooveEngine = std::make_unique<GrooveEngine>(grooveConfig);
        
        DiagnosticsEngine::Config diagConfig;
        diagConfig.sampleRate = sr;
        diagnosticsEngine = std::make_unique<DiagnosticsEngine>(diagConfig);
        
        initialized = true;
        LOGI("Penta Core instance created with sample rate: %.1f", sr);
    }
    
    ~PentaCoreInstance() {
        initialized = false;
        LOGI("Penta Core instance destroyed");
    }
};

// Global instance (managed by JNI)
static std::unique_ptr<PentaCoreInstance> g_pentaCore;
static std::mutex g_instanceMutex;

// Forward declarations for JNI bridge
extern "C" {

// Initialize Penta Core instance
void pentaCoreInitialize(double sampleRate) {
    std::lock_guard<std::mutex> lock(g_instanceMutex);
    if (g_pentaCore == nullptr) {
        g_pentaCore = std::make_unique<PentaCoreInstance>(sampleRate);
        LOGI("Penta Core initialized");
    } else {
        LOGI("Penta Core already initialized");
    }
}

// Cleanup Penta Core instance
void pentaCoreCleanup() {
    std::lock_guard<std::mutex> lock(g_instanceMutex);
    g_pentaCore.reset();
    LOGI("Penta Core cleaned up");
}

// Handle MIDI message from JUCE bridge
void pentaCoreHandleMidiMessage(const uint8_t* data, size_t length, long timestamp) {
    if (data == nullptr || length == 0) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(g_instanceMutex);
    if (g_pentaCore == nullptr || !g_pentaCore->initialized) {
        return;
    }
    
    // Parse MIDI message
    uint8_t status = data[0] & 0xF0;
    uint8_t channel = data[0] & 0x0F;
    
    if (status == 0x90 && length >= 3) {
        // Note On
        uint8_t note = data[1];
        uint8_t velocity = data[2];
        
        if (velocity > 0) {
            Note midiNote;
            midiNote.pitch = note;
            midiNote.velocity = velocity;
            midiNote.timestamp = timestamp;
            
            {
                std::lock_guard<std::mutex> notesLock(g_pentaCore->notesMutex);
                g_pentaCore->activeNotes.push_back(midiNote);
            }
            
            // Process note through harmony engine
            std::vector<Note> notes;
            notes.push_back(midiNote);
            g_pentaCore->harmonyEngine->processNotes(notes.data(), notes.size());
            
            LOGI("MIDI Note On: note=%d, velocity=%d", note, velocity);
        } else {
            // Note Off (velocity 0)
            // Remove note from active notes
            std::lock_guard<std::mutex> notesLock(g_pentaCore->notesMutex);
            g_pentaCore->activeNotes.erase(
                std::remove_if(
                    g_pentaCore->activeNotes.begin(),
                    g_pentaCore->activeNotes.end(),
                    [note](const Note& n) { return n.pitch == note; }
                ),
                g_pentaCore->activeNotes.end()
            );
            
            LOGI("MIDI Note Off: note=%d", note);
        }
    } else if (status == 0x80 && length >= 3) {
        // Note Off
        uint8_t note = data[1];
        
        std::lock_guard<std::mutex> notesLock(g_pentaCore->notesMutex);
        g_pentaCore->activeNotes.erase(
            std::remove_if(
                g_pentaCore->activeNotes.begin(),
                g_pentaCore->activeNotes.end(),
                [note](const Note& n) { return n.pitch == note; }
            ),
            g_pentaCore->activeNotes.end()
        );
        
        LOGI("MIDI Note Off: note=%d", note);
    }
}

// Handle MIDI device changes
void pentaCoreOnMidiDevicesChanged() {
    LOGI("MIDI devices changed - refreshing device list");
    // Could trigger device list refresh in UI
}

// Process audio through Penta Core (called from Oboe callback)
void pentaCoreProcessAudio(float* inputL, float* inputR,
                          float* outputL, float* outputR,
                          int32_t frameCount) {
    std::lock_guard<std::mutex> lock(g_instanceMutex);
    if (g_pentaCore == nullptr || !g_pentaCore->initialized) {
        // Pass-through if not initialized
        if (inputL != nullptr && outputL != nullptr) {
            std::memcpy(outputL, inputL, frameCount * sizeof(float));
        }
        if (inputR != nullptr && outputR != nullptr) {
            std::memcpy(outputR, inputR, frameCount * sizeof(float));
        }
        return;
    }
    
    // Process audio through diagnostics engine
    if (g_pentaCore->diagnosticsEngine != nullptr && inputL != nullptr) {
        // Analyze audio for diagnostics
        // Note: DiagnosticsEngine may need audio buffer interface
    }
    
    // For now, pass-through audio
    // In the future, this could apply harmony/groove processing
    if (inputL != nullptr && outputL != nullptr) {
        std::memcpy(outputL, inputL, frameCount * sizeof(float));
    }
    if (inputR != nullptr && outputR != nullptr) {
        std::memcpy(outputR, inputR, frameCount * sizeof(float));
    }
}

// Get current chord (for UI display)
const char* pentaCoreGetCurrentChord() {
    std::lock_guard<std::mutex> lock(g_instanceMutex);
    if (g_pentaCore == nullptr || !g_pentaCore->initialized) {
        return "N/A";
    }
    
    const Chord& chord = g_pentaCore->harmonyEngine->getCurrentChord();
    // Convert chord to string representation
    // This is a simplified version - full implementation would format chord name
    static char chordStr[32];
    snprintf(chordStr, sizeof(chordStr), "Chord_%d", static_cast<int>(chord.root));
    return chordStr;
}

// Get current scale (for UI display)
const char* pentaCoreGetCurrentScale() {
    std::lock_guard<std::mutex> lock(g_instanceMutex);
    if (g_pentaCore == nullptr || !g_pentaCore->initialized) {
        return "N/A";
    }
    
    const Scale& scale = g_pentaCore->harmonyEngine->getCurrentScale();
    // Convert scale to string representation
    static char scaleStr[32];
    snprintf(scaleStr, sizeof(scaleStr), "Scale_%d", static_cast<int>(scale.root));
    return scaleStr;
}

// Get detected tempo (for UI display)
float pentaCoreGetDetectedTempo() {
    std::lock_guard<std::mutex> lock(g_instanceMutex);
    if (g_pentaCore == nullptr || !g_pentaCore->initialized) {
        return 0.0f;
    }
    
    // Get tempo from groove engine
    // Note: GrooveEngine may need tempo getter method
    return 120.0f; // Placeholder
}

//==============================================================================
// JNI methods for PentaCoreNative.kt
//==============================================================================

JNIEXPORT void JNICALL
Java_com_idaw_jni_PentaCoreNative_initialize(
    JNIEnv* env,
    jclass clazz,
    jdouble sampleRate)
{
    pentaCoreInitialize(static_cast<double>(sampleRate));
}

JNIEXPORT void JNICALL
Java_com_idaw_jni_PentaCoreNative_cleanup(
    JNIEnv* env,
    jclass clazz)
{
    pentaCoreCleanup();
}

JNIEXPORT void JNICALL
Java_com_idaw_jni_PentaCoreNative_handleMidiMessage(
    JNIEnv* env,
    jclass clazz,
    jbyteArray data,
    jint offset,
    jint count,
    jlong timestamp)
{
    if (data == nullptr || count <= 0) {
        return;
    }
    
    jbyte* bytes = env->GetByteArrayElements(data, nullptr);
    if (bytes != nullptr) {
        pentaCoreHandleMidiMessage(
            reinterpret_cast<const uint8_t*>(bytes + offset),
            static_cast<size_t>(count),
            timestamp
        );
        env->ReleaseByteArrayElements(data, bytes, JNI_ABORT);
    }
}

JNIEXPORT void JNICALL
Java_com_idaw_jni_PentaCoreNative_processAudio(
    JNIEnv* env,
    jclass clazz,
    jfloatArray inputL,
    jfloatArray inputR,
    jfloatArray outputL,
    jfloatArray outputR,
    jint frameCount)
{
    if (outputL == nullptr || outputR == nullptr || frameCount <= 0) {
        return;
    }
    
    jfloat* inL = inputL != nullptr ? env->GetFloatArrayElements(inputL, nullptr) : nullptr;
    jfloat* inR = inputR != nullptr ? env->GetFloatArrayElements(inputR, nullptr) : nullptr;
    jfloat* outL = env->GetFloatArrayElements(outputL, nullptr);
    jfloat* outR = env->GetFloatArrayElements(outputR, nullptr);
    
    if (outL != nullptr && outR != nullptr) {
        pentaCoreProcessAudio(inL, inR, outL, outR, frameCount);
    }
    
    if (inL != nullptr) env->ReleaseFloatArrayElements(inputL, inL, JNI_ABORT);
    if (inR != nullptr) env->ReleaseFloatArrayElements(inputR, inR, JNI_ABORT);
    env->ReleaseFloatArrayElements(outputL, outL, 0);
    env->ReleaseFloatArrayElements(outputR, outR, 0);
}

JNIEXPORT jstring JNICALL
Java_com_idaw_jni_PentaCoreNative_getCurrentChord(
    JNIEnv* env,
    jclass clazz)
{
    const char* chord = pentaCoreGetCurrentChord();
    return env->NewStringUTF(chord);
}

JNIEXPORT jstring JNICALL
Java_com_idaw_jni_PentaCoreNative_getCurrentScale(
    JNIEnv* env,
    jclass clazz)
{
    const char* scale = pentaCoreGetCurrentScale();
    return env->NewStringUTF(scale);
}

JNIEXPORT jfloat JNICALL
Java_com_idaw_jni_PentaCoreNative_getDetectedTempo(
    JNIEnv* env,
    jclass clazz)
{
    return pentaCoreGetDetectedTempo();
}

} // extern "C"

