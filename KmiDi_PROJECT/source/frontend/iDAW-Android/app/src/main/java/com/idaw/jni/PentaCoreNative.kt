// PentaCoreNative.kt
// JNI declarations for Penta Core native methods

package com.idaw.jni

object PentaCoreNative {
    init {
        System.loadLibrary("idaw_android")
    }

    // Initialization
    external fun initialize(sampleRate: Double)
    external fun cleanup()

    // MIDI handling (called from JUCE bridge)
    external fun handleMidiMessage(data: ByteArray, offset: Int, count: Int, timestamp: Long)

    // Audio processing (called from Oboe callback)
    external fun processAudio(
        inputL: FloatArray?,
        inputR: FloatArray?,
        outputL: FloatArray,
        outputR: FloatArray,
        frameCount: Int
    )

    // State queries (for UI)
    external fun getCurrentChord(): String
    external fun getCurrentScale(): String
    external fun getDetectedTempo(): Float
}

