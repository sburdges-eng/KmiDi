use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int};
use std::ptr;
use std::sync::{Arc, Mutex, OnceLock};
use serde::{Deserialize, Serialize};
use serde_json;

// =============================================================================
// FFI Error Codes (must match C header)
// =============================================================================

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KellyErrorCode {
    Success = 0,
    NullPointer = -1,
    InitializationFailed = -2,
    InvalidParameter = -3,
    JsonParseError = -4,
    MemoryAllocation = -5,
    FileNotFound = -6,
    Unknown = -999,
}

// =============================================================================
// Rust Data Types
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentResult {
    pub core_wound: String,
    pub core_desire: String,
    pub emotional_intent: String,
    pub valence: f32,
    pub arousal: f32,
    pub dominance: f32,
    pub complexity: f32,
    pub progression: Vec<String>,
    pub key: String,
    pub bpm: i32,
    pub genre: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MidiEvent {
    pub event_type: i32,
    pub timestamp: f64,
    pub note: u8,
    pub velocity: u8,
    pub duration: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MidiTrack {
    pub name: String,
    pub channel: i32,
    pub events: Vec<MidiEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedMidi {
    pub bars: i32,
    pub bpm: i32,
    pub key: String,
    pub time_signature: String,
    pub tracks: Vec<MidiTrack>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionState {
    pub valence: f32,
    pub arousal: f32,
    pub dominance: f32,
    pub complexity: f32,
}

// =============================================================================
// FFI Declarations (external C functions)
// =============================================================================

#[link(name = "KellyFFI")]
extern "C" {
    // Error handling
    fn kelly_get_error_message(error_code: KellyErrorCode) -> *const c_char;
    fn kelly_get_last_error() -> *const c_char;
    
    // Memory management
    fn kelly_free_string(ptr: *mut c_char);
    
    // Core functions
    fn kelly_brain_create() -> *mut KellyBrainHandle;
    fn kelly_brain_initialize(brain: *mut KellyBrainHandle, data_path: *const c_char) -> KellyErrorCode;
    fn kelly_brain_is_initialized(brain: *const KellyBrainHandle) -> bool;
    fn kelly_brain_destroy(brain: *mut KellyBrainHandle);
    
    // Intent processing
    fn kelly_brain_from_text(brain: *mut KellyBrainHandle, text: *const c_char) -> *mut c_char;
    fn kelly_brain_from_emotion(brain: *mut KellyBrainHandle, emotion_name: *const c_char, intensity: c_float) -> *mut c_char;
    fn kelly_brain_from_journey(brain: *mut KellyBrainHandle, current_state: *const c_char, desired_state: *const c_char) -> *mut c_char;
    fn kelly_brain_from_wound(brain: *mut KellyBrainHandle, wound_json: *const c_char) -> *mut c_char;
    
    // MIDI generation
    fn kelly_brain_generate_midi(brain: *mut KellyBrainHandle, intent_json: *const c_char, bars: c_int) -> *mut c_char;
    fn kelly_brain_generate_midi_with_params(brain: *mut KellyBrainHandle, intent_json: *const c_char, 
                                           bars: c_int, bpm: c_int, key_signature: *const c_char) -> *mut c_char;
    
    // State queries
    fn kelly_brain_get_emotion_state(brain: *const KellyBrainHandle) -> *mut c_char;
    fn kelly_brain_get_parameters(brain: *const KellyBrainHandle) -> *mut c_char;
    fn kelly_brain_get_available_emotions(brain: *const KellyBrainHandle) -> *mut c_char;
    
    // Parameter updates
    fn kelly_brain_set_emotion_parameters(brain: *mut KellyBrainHandle, valence: c_float, arousal: c_float, dominance: c_float) -> KellyErrorCode;
    fn kelly_brain_update_parameters(brain: *mut KellyBrainHandle, params_json: *const c_char) -> KellyErrorCode;
    
    // Utility functions
    fn kelly_get_version() -> *const c_char;
    fn kelly_check_data_files(data_path: *const c_char) -> bool;
}

// Opaque handle type (matches C typedef)
#[repr(C)]
pub struct KellyBrainHandle {
    _private: [u8; 0],
}

// =============================================================================
// Error Handling
// =============================================================================

#[derive(Debug, Clone)]
pub struct KellyError {
    pub code: KellyErrorCode,
    pub message: String,
}

impl std::fmt::Display for KellyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "KellyError({:?}): {}", self.code, self.message)
    }
}

impl std::error::Error for KellyError {}

pub type KellyResult<T> = Result<T, KellyError>;

impl From<KellyErrorCode> for KellyError {
    fn from(code: KellyErrorCode) -> Self {
        let message = unsafe {
            let msg_ptr = kelly_get_error_message(code);
            if msg_ptr.is_null() {
                "Unknown error".to_string()
            } else {
                CStr::from_ptr(msg_ptr).to_string_lossy().to_string()
            }
        };
        
        KellyError { code, message }
    }
}

// =============================================================================
// Safe Rust Wrapper
// =============================================================================

pub struct KellyBrain {
    handle: *mut KellyBrainHandle,
}

unsafe impl Send for KellyBrain {}
unsafe impl Sync for KellyBrain {}

impl KellyBrain {
    /// Create a new KellyBrain instance
    pub fn new() -> KellyResult<Self> {
        let handle = unsafe { kelly_brain_create() };
        if handle.is_null() {
            return Err(KellyError {
                code: KellyErrorCode::MemoryAllocation,
                message: "Failed to create KellyBrain instance".to_string(),
            });
        }
        
        Ok(KellyBrain { handle })
    }
    
    /// Initialize KellyBrain with data directory
    pub fn initialize(&mut self, data_path: &str) -> KellyResult<()> {
        let c_path = CString::new(data_path)
            .map_err(|_| KellyError {
                code: KellyErrorCode::InvalidParameter,
                message: "Invalid data path".to_string(),
            })?;
        
        let result = unsafe { kelly_brain_initialize(self.handle, c_path.as_ptr()) };
        match result {
            KellyErrorCode::Success => Ok(()),
            error => Err(error.into()),
        }
    }
    
    /// Check if KellyBrain is initialized
    pub fn is_initialized(&self) -> bool {
        unsafe { kelly_brain_is_initialized(self.handle) }
    }
    
    /// Process text and generate intent result
    pub fn from_text(&mut self, text: &str) -> KellyResult<IntentResult> {
        let c_text = CString::new(text)
            .map_err(|_| KellyError {
                code: KellyErrorCode::InvalidParameter,
                message: "Invalid text parameter".to_string(),
            })?;
        
        let json_ptr = unsafe { kelly_brain_from_text(self.handle, c_text.as_ptr()) };
        if json_ptr.is_null() {
            return Err(self.get_last_error());
        }
        
        let json_str = unsafe {
            let json_cstr = CStr::from_ptr(json_ptr);
            let result = json_cstr.to_string_lossy().to_string();
            kelly_free_string(json_ptr);
            result
        };
        
        serde_json::from_str(&json_str)
            .map_err(|e| KellyError {
                code: KellyErrorCode::JsonParseError,
                message: format!("JSON parse error: {}", e),
            })
    }
    
    /// Process emotion name and generate intent result
    pub fn from_emotion(&mut self, emotion_name: &str, intensity: f32) -> KellyResult<IntentResult> {
        let c_emotion = CString::new(emotion_name)
            .map_err(|_| KellyError {
                code: KellyErrorCode::InvalidParameter,
                message: "Invalid emotion name".to_string(),
            })?;
        
        if !(0.0..=1.0).contains(&intensity) {
            return Err(KellyError {
                code: KellyErrorCode::InvalidParameter,
                message: "Intensity must be between 0.0 and 1.0".to_string(),
            });
        }
        
        let json_ptr = unsafe { 
            kelly_brain_from_emotion(self.handle, c_emotion.as_ptr(), intensity as c_float) 
        };
        if json_ptr.is_null() {
            return Err(self.get_last_error());
        }
        
        let json_str = unsafe {
            let json_cstr = CStr::from_ptr(json_ptr);
            let result = json_cstr.to_string_lossy().to_string();
            kelly_free_string(json_ptr);
            result
        };
        
        serde_json::from_str(&json_str)
            .map_err(|e| KellyError {
                code: KellyErrorCode::JsonParseError,
                message: format!("JSON parse error: {}", e),
            })
    }
    
    /// Generate MIDI from intent result
    pub fn generate_midi(&mut self, intent: &IntentResult, bars: i32) -> KellyResult<GeneratedMidi> {
        if bars <= 0 || bars > 64 {
            return Err(KellyError {
                code: KellyErrorCode::InvalidParameter,
                message: "Bars must be between 1 and 64".to_string(),
            });
        }
        
        let intent_json = serde_json::to_string(intent)
            .map_err(|e| KellyError {
                code: KellyErrorCode::JsonParseError,
                message: format!("Intent serialization error: {}", e),
            })?;
        
        let c_intent = CString::new(intent_json)
            .map_err(|_| KellyError {
                code: KellyErrorCode::InvalidParameter,
                message: "Invalid intent JSON".to_string(),
            })?;
        
        let json_ptr = unsafe { 
            kelly_brain_generate_midi(self.handle, c_intent.as_ptr(), bars as c_int) 
        };
        if json_ptr.is_null() {
            return Err(self.get_last_error());
        }
        
        let json_str = unsafe {
            let json_cstr = CStr::from_ptr(json_ptr);
            let result = json_cstr.to_string_lossy().to_string();
            kelly_free_string(json_ptr);
            result
        };
        
        serde_json::from_str(&json_str)
            .map_err(|e| KellyError {
                code: KellyErrorCode::JsonParseError,
                message: format!("MIDI JSON parse error: {}", e),
            })
    }
    
    /// Generate MIDI with specific parameters
    pub fn generate_midi_with_params(
        &mut self, 
        intent: &IntentResult, 
        bars: i32, 
        bpm: i32, 
        key_signature: &str
    ) -> KellyResult<GeneratedMidi> {
        if bars <= 0 || bars > 64 {
            return Err(KellyError {
                code: KellyErrorCode::InvalidParameter,
                message: "Bars must be between 1 and 64".to_string(),
            });
        }
        
        if bpm <= 0 || bpm > 300 {
            return Err(KellyError {
                code: KellyErrorCode::InvalidParameter,
                message: "BPM must be between 1 and 300".to_string(),
            });
        }
        
        let intent_json = serde_json::to_string(intent)
            .map_err(|e| KellyError {
                code: KellyErrorCode::JsonParseError,
                message: format!("Intent serialization error: {}", e),
            })?;
        
        let c_intent = CString::new(intent_json)
            .map_err(|_| KellyError {
                code: KellyErrorCode::InvalidParameter,
                message: "Invalid intent JSON".to_string(),
            })?;
        
        let c_key = CString::new(key_signature)
            .map_err(|_| KellyError {
                code: KellyErrorCode::InvalidParameter,
                message: "Invalid key signature".to_string(),
            })?;
        
        let json_ptr = unsafe { 
            kelly_brain_generate_midi_with_params(
                self.handle, 
                c_intent.as_ptr(), 
                bars as c_int, 
                bpm as c_int, 
                c_key.as_ptr()
            ) 
        };
        if json_ptr.is_null() {
            return Err(self.get_last_error());
        }
        
        let json_str = unsafe {
            let json_cstr = CStr::from_ptr(json_ptr);
            let result = json_cstr.to_string_lossy().to_string();
            kelly_free_string(json_ptr);
            result
        };
        
        serde_json::from_str(&json_str)
            .map_err(|e| KellyError {
                code: KellyErrorCode::JsonParseError,
                message: format!("MIDI JSON parse error: {}", e),
            })
    }
    
    /// Get current emotion state
    pub fn get_emotion_state(&self) -> KellyResult<EmotionState> {
        let json_ptr = unsafe { kelly_brain_get_emotion_state(self.handle) };
        if json_ptr.is_null() {
            return Err(self.get_last_error());
        }
        
        let json_str = unsafe {
            let json_cstr = CStr::from_ptr(json_ptr);
            let result = json_cstr.to_string_lossy().to_string();
            kelly_free_string(json_ptr);
            result
        };
        
        serde_json::from_str(&json_str)
            .map_err(|e| KellyError {
                code: KellyErrorCode::JsonParseError,
                message: format!("Emotion state JSON parse error: {}", e),
            })
    }
    
    /// Set emotion parameters
    pub fn set_emotion_parameters(&mut self, valence: f32, arousal: f32, dominance: f32) -> KellyResult<()> {
        if !(-1.0..=1.0).contains(&valence) {
            return Err(KellyError {
                code: KellyErrorCode::InvalidParameter,
                message: "Valence must be between -1.0 and 1.0".to_string(),
            });
        }
        
        if !(0.0..=1.0).contains(&arousal) || !(0.0..=1.0).contains(&dominance) {
            return Err(KellyError {
                code: KellyErrorCode::InvalidParameter,
                message: "Arousal and dominance must be between 0.0 and 1.0".to_string(),
            });
        }
        
        let result = unsafe { 
            kelly_brain_set_emotion_parameters(
                self.handle, 
                valence as c_float, 
                arousal as c_float, 
                dominance as c_float
            ) 
        };
        
        match result {
            KellyErrorCode::Success => Ok(()),
            error => Err(error.into()),
        }
    }
    
    /// Get available emotions
    pub fn get_available_emotions(&self) -> KellyResult<serde_json::Value> {
        let json_ptr = unsafe { kelly_brain_get_available_emotions(self.handle) };
        if json_ptr.is_null() {
            return Err(self.get_last_error());
        }
        
        let json_str = unsafe {
            let json_cstr = CStr::from_ptr(json_ptr);
            let result = json_cstr.to_string_lossy().to_string();
            kelly_free_string(json_ptr);
            result
        };
        
        serde_json::from_str(&json_str)
            .map_err(|e| KellyError {
                code: KellyErrorCode::JsonParseError,
                message: format!("Emotions JSON parse error: {}", e),
            })
    }
    
    /// Get version information
    pub fn get_version() -> String {
        unsafe {
            let version_ptr = kelly_get_version();
            if version_ptr.is_null() {
                "Unknown version".to_string()
            } else {
                CStr::from_ptr(version_ptr).to_string_lossy().to_string()
            }
        }
    }
    
    /// Check if data files are available
    pub fn check_data_files(data_path: &str) -> bool {
        let c_path = match CString::new(data_path) {
            Ok(path) => path,
            Err(_) => return false,
        };
        
        unsafe { kelly_check_data_files(c_path.as_ptr()) }
    }
    
    /// Get last error from C library
    fn get_last_error(&self) -> KellyError {
        unsafe {
            let error_ptr = kelly_get_last_error();
            let message = if error_ptr.is_null() {
                "Unknown error occurred".to_string()
            } else {
                CStr::from_ptr(error_ptr).to_string_lossy().to_string()
            };
            
            KellyError {
                code: KellyErrorCode::Unknown,
                message,
            }
        }
    }
}

impl Drop for KellyBrain {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { kelly_brain_destroy(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

// =============================================================================
// Global KellyBrain Instance Management
// =============================================================================

pub struct KellyBrainManager {
    brain: Arc<Mutex<Option<KellyBrain>>>,
}

impl KellyBrainManager {
    pub fn new() -> Self {
        KellyBrainManager {
            brain: Arc::new(Mutex::new(None)),
        }
    }
    
    /// Initialize global KellyBrain instance
    pub fn initialize(&self, data_path: &str) -> KellyResult<()> {
        let mut brain_guard = self.brain.lock().unwrap();
        
        let mut brain = KellyBrain::new()?;
        brain.initialize(data_path)?;
        
        *brain_guard = Some(brain);
        Ok(())
    }
    
    /// Execute operation with KellyBrain instance
    pub fn with_brain<T, F>(&self, f: F) -> KellyResult<T>
    where
        F: FnOnce(&mut KellyBrain) -> KellyResult<T>,
    {
        let mut brain_guard = self.brain.lock().unwrap();
        match brain_guard.as_mut() {
            Some(brain) => f(brain),
            None => Err(KellyError {
                code: KellyErrorCode::InitializationFailed,
                message: "KellyBrain not initialized".to_string(),
            }),
        }
    }
    
    /// Check if KellyBrain is initialized
    pub fn is_initialized(&self) -> bool {
        let brain_guard = self.brain.lock().unwrap();
        brain_guard.as_ref().map_or(false, |brain| brain.is_initialized())
    }
}

static KELLY_BRAIN_MANAGER: OnceLock<KellyBrainManager> = OnceLock::new();

/// Get global KellyBrain manager instance
pub fn get_kelly_brain_manager() -> &'static KellyBrainManager {
    KELLY_BRAIN_MANAGER.get_or_init(|| KellyBrainManager::new())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_kelly_brain_creation() {
        let brain_result = KellyBrain::new();
        assert!(brain_result.is_ok());
    }
    
    #[test]
    fn test_version() {
        let version = KellyBrain::get_version();
        assert!(!version.is_empty());
        assert!(version.contains("KellyBrain"));
    }
    
    #[test]
    fn test_error_conversion() {
        let error: KellyError = KellyErrorCode::InvalidParameter.into();
        assert_eq!(error.code, KellyErrorCode::InvalidParameter);
        assert!(!error.message.is_empty());
    }
}