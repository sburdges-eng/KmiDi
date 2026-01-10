// iDAWApplication.kt
// Application class for iDAW Android

package com.idaw

import android.app.Application

class iDAWApplication : Application() {
    
    override fun onCreate() {
        super.onCreate()
        android.util.Log.i(TAG, "iDAW Application created")
    }
    
    companion object {
        private const val TAG = "iDAWApplication"
    }
}

