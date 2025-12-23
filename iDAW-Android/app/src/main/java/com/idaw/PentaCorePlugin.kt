// PentaCorePlugin.kt
// Penta Core plugin wrapper for Android

package com.idaw

import android.content.Context
import com.rmsl.juce.JuceMidiSupport
import com.idaw.jni.PentaCoreNative

class PentaCorePlugin(private val context: Context) {
    
    private var midiDeviceManager: JuceMidiSupport.MidiDeviceManager? = null
    private var bluetoothMidiManager: JuceMidiSupport.BluetoothMidiManager? = null
    private var initialized = false
    
    fun initialize(sampleRate: Double = 48000.0) {
        if (initialized) {
            return
        }
        
        // Initialize native Penta Core
        PentaCoreNative.initialize(sampleRate)
        
        // Initialize MIDI device manager
        midiDeviceManager = JuceMidiSupport.getAndroidMidiDeviceManager(context)
        if (midiDeviceManager == null) {
            android.util.Log.w(TAG, "MIDI device manager not available")
        }
        
        // Initialize Bluetooth MIDI manager
        bluetoothMidiManager = JuceMidiSupport.getAndroidBluetoothManager(context)
        if (bluetoothMidiManager == null) {
            android.util.Log.w(TAG, "Bluetooth MIDI manager not available")
        }
        
        initialized = true
        android.util.Log.i(TAG, "Penta Core plugin initialized")
    }
    
    fun cleanup() {
        if (!initialized) {
            return
        }
        
        // Cleanup native Penta Core
        PentaCoreNative.cleanup()
        
        // MIDI managers are cleaned up automatically by JUCE
        midiDeviceManager = null
        bluetoothMidiManager = null
        
        initialized = false
        android.util.Log.i(TAG, "Penta Core plugin cleaned up")
    }
    
    fun getMidiInputDevices(): Array<String> {
        return midiDeviceManager?.getJuceAndroidMidiInputDeviceNameAndIDs() ?: emptyArray()
    }
    
    fun getMidiOutputDevices(): Array<String> {
        return midiDeviceManager?.getJuceAndroidMidiOutputDeviceNameAndIDs() ?: emptyArray()
    }
    
    fun openMidiInputPort(deviceID: Int, host: Long): JuceMidiSupport.JuceMidiPort? {
        return midiDeviceManager?.openMidiInputPortWithID(deviceID, host)
    }
    
    fun openMidiOutputPort(deviceID: Int): JuceMidiSupport.JuceMidiPort? {
        return midiDeviceManager?.openMidiOutputPortWithID(deviceID)
    }
    
    fun getCurrentChord(): String {
        return if (initialized) PentaCoreNative.getCurrentChord() else "N/A"
    }
    
    fun getCurrentScale(): String {
        return if (initialized) PentaCoreNative.getCurrentScale() else "N/A"
    }
    
    fun getDetectedTempo(): Float {
        return if (initialized) PentaCoreNative.getDetectedTempo() else 0.0f
    }
    
    companion object {
        private const val TAG = "PentaCorePlugin"
    }
}

