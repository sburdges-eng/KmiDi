// MainActivity.kt
// Main activity for iDAW Android application

package com.idaw

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioManager
import android.os.Build
import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.rmsl.juce.FragmentOverlay

class MainActivity : AppCompatActivity() {
    
    private lateinit var plugin: PentaCorePlugin
    private lateinit var chordDisplay: TextView
    private lateinit var scaleDisplay: TextView
    private lateinit var tempoDisplay: TextView
    
    private val updateDisplayRunnable = object : Runnable {
        override fun run() {
            updateDisplay()
            chordDisplay.postDelayed(this, 100) // Update every 100ms
        }
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Set up audio
        audioManager = getSystemService(AUDIO_SERVICE) as AudioManager
        audioManager?.mode = AudioManager.MODE_NORMAL
        
        // Request permissions
        requestPermissions()
        
        // Initialize plugin
        plugin = PentaCorePlugin(this)
        plugin.initialize(48000.0)
        
        // Create simple UI
        setupUI()
        
        // Start display updates
        chordDisplay.post(updateDisplayRunnable)
    }
    
    override fun onDestroy() {
        super.onDestroy()
        chordDisplay.removeCallbacks(updateDisplayRunnable)
        plugin.cleanup()
    }
    
    private fun setupUI() {
        // Create a simple text-based UI
        // In a real implementation, this would use proper layouts
        chordDisplay = TextView(this).apply {
            text = "Chord: ${plugin.getCurrentChord()}"
            textSize = 18f
        }
        
        scaleDisplay = TextView(this).apply {
            text = "Scale: ${plugin.getCurrentScale()}"
            textSize = 18f
        }
        
        tempoDisplay = TextView(this).apply {
            text = "Tempo: ${plugin.getDetectedTempo().toInt()} BPM"
            textSize = 18f
        }
        
        // For now, just log - proper UI would use setContentView with a layout
        android.util.Log.i(TAG, "UI setup complete")
    }
    
    private fun updateDisplay() {
        if (::chordDisplay.isInitialized) {
            chordDisplay.text = "Chord: ${plugin.getCurrentChord()}"
            scaleDisplay.text = "Scale: ${plugin.getCurrentScale()}"
            tempoDisplay.text = "Tempo: ${plugin.getDetectedTempo().toInt()} BPM"
        }
    }
    
    private fun requestPermissions() {
        val permissions = mutableListOf<String>()
        
        // Audio permissions
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED) {
            permissions.add(Manifest.permission.RECORD_AUDIO)
        }
        
        // Bluetooth permissions (Android 12+)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.BLUETOOTH_CONNECT)
                != PackageManager.PERMISSION_GRANTED) {
                permissions.add(Manifest.permission.BLUETOOTH_CONNECT)
            }
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.BLUETOOTH_SCAN)
                != PackageManager.PERMISSION_GRANTED) {
                permissions.add(Manifest.permission.BLUETOOTH_SCAN)
            }
        }
        
        if (permissions.isNotEmpty()) {
            ActivityCompat.requestPermissions(
                this,
                permissions.toTypedArray(),
                PERMISSION_REQUEST_CODE
            )
        }
    }
    
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        
        if (requestCode == PERMISSION_REQUEST_CODE) {
            for (i in permissions.indices) {
                if (grantResults[i] == PackageManager.PERMISSION_GRANTED) {
                    android.util.Log.i(TAG, "Permission granted: ${permissions[i]}")
                } else {
                    android.util.Log.w(TAG, "Permission denied: ${permissions[i]}")
                }
            }
        }
    }
    
    companion object {
        private const val TAG = "MainActivity"
        private const val PERMISSION_REQUEST_CODE = 1001
    }
    
    private var audioManager: AudioManager? = null
}

