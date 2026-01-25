//! Intent IR Manager - Owns all IntentFrame construction and validation
//!
//! This module provides the IntentIRManager which:
//! - Owns all IntentFrame construction
//! - Validates all incoming IR
//! - Handles version negotiation
//! - Clamps values to valid ranges
//! - Provides snapshots to C++

use super::*;
use super::validator::{validate_and_clamp, clamp_frame, ValidationError};
use super::ffi::{to_c_frame, CIntentFrame};
use std::sync::{Arc, Mutex, RwLock};
use std::sync::atomic::{AtomicU64, Ordering};

/// Intent IR Manager - Single source of truth for IntentFrame
pub struct IntentIRManager {
    /// Current validated IntentFrame snapshot
    current_frame: Arc<RwLock<IntentFrame>>,
    /// Session ID counter
    session_counter: AtomicU64,
    /// Intent ID counter
    intent_counter: AtomicU64,
}

impl IntentIRManager {
    /// Create a new Intent IR Manager
    pub fn new() -> Self {
        Self {
            current_frame: Arc::new(RwLock::new(IntentFrame::default())),
            session_counter: AtomicU64::new(1),
            intent_counter: AtomicU64::new(1),
        }
    }
    
    /// Validate and store an IntentFrame
    /// 
    /// This is the only way to update the current IntentFrame.
    /// All incoming IR must go through this method.
    pub fn validate_and_store(&self, frame: IntentFrame) -> Result<(), ValidationError> {
        // Validate the frame
        let validated = validate_and_clamp(frame)?;
        
        // Update current frame
        let mut current = self.current_frame.write().unwrap();
        *current = validated;
        
        Ok(())
    }
    
    /// Validate and store with clamping (always succeeds)
    /// 
    /// Use this when you want to accept invalid input and clamp it to valid ranges.
    pub fn clamp_and_store(&self, frame: IntentFrame) {
        let clamped = clamp_frame(frame);
        let mut current = self.current_frame.write().unwrap();
        *current = clamped;
    }
    
    /// Get current IntentFrame snapshot (read-only)
    /// 
    /// This provides a copy of the current frame that can be safely read
    /// from any thread, including the audio thread.
    pub fn get_snapshot(&self) -> IntentFrame {
        let current = self.current_frame.read().unwrap();
        current.clone()
    }
    
    /// Get current IntentFrame as C struct (for FFI)
    /// 
    /// This provides a C-compatible struct that can be passed to C++.
    pub fn get_c_snapshot(&self) -> CIntentFrame {
        let current = self.current_frame.read().unwrap();
        to_c_frame(&current)
    }
    
    /// Generate a new session ID
    pub fn new_session_id(&self) -> u64 {
        self.session_counter.fetch_add(1, Ordering::Relaxed)
    }
    
    /// Generate a new intent ID
    pub fn new_intent_id(&self) -> u64 {
        self.intent_counter.fetch_add(1, Ordering::Relaxed)
    }
    
    /// Check if current frame is valid (non-zero intent_id)
    pub fn is_valid(&self) -> bool {
        let current = self.current_frame.read().unwrap();
        current.meta.intent_id != 0
    }
    
    /// Get current frame's intent ID
    pub fn get_current_intent_id(&self) -> u64 {
        let current = self.current_frame.read().unwrap();
        current.meta.intent_id
    }
    
    /// Get current frame's session ID
    pub fn get_current_session_id(&self) -> u64 {
        let current = self.current_frame.read().unwrap();
        current.meta.session_id
    }
    
    /// Update emotion state (convenience method)
    pub fn update_emotion(
        &self,
        valence: f32,
        arousal: f32,
        dominance: f32,
        discrete_id: Option<i16>,
        intensity: f32,
        confidence: f32,
    ) -> Result<(), ValidationError> {
        let mut frame = self.get_snapshot();
        frame.emotion.valence = valence;
        frame.emotion.arousal = arousal;
        frame.emotion.dominance = dominance;
        frame.emotion.discrete_id = discrete_id.unwrap_or(-1);
        frame.emotion.intensity = intensity;
        frame.emotion.confidence = confidence;
        
        // Update intent ID to mark as new
        frame.meta.intent_id = self.new_intent_id();
        
        self.validate_and_store(frame)
    }
    
    /// Update musical intent (convenience method)
    pub fn update_music(
        &self,
        tempo_bias: f32,
        rhythmic_density: f32,
        groove_strength: f32,
        harmonic_tension: f32,
        harmonic_motion: f32,
        mode_preference: i8,
        melodic_activity: f32,
        contour_variance: f32,
        dynamic_range: f32,
        texture_density: f32,
    ) -> Result<(), ValidationError> {
        let mut frame = self.get_snapshot();
        frame.music.tempo_bias = tempo_bias;
        frame.music.rhythmic_density = rhythmic_density;
        frame.music.groove_strength = groove_strength;
        frame.music.harmonic_tension = harmonic_tension;
        frame.music.harmonic_motion = harmonic_motion;
        frame.music.mode_preference = mode_preference;
        frame.music.melodic_activity = melodic_activity;
        frame.music.contour_variance = contour_variance;
        frame.music.dynamic_range = dynamic_range;
        frame.music.texture_density = texture_density;
        
        // Update intent ID to mark as new
        frame.meta.intent_id = self.new_intent_id();
        
        self.validate_and_store(frame)
    }
    
    /// Update time scope (convenience method)
    pub fn update_time(
        &self,
        start_bar: i32,
        end_bar: i32,
        fade_in_beats: f32,
        fade_out_beats: f32,
    ) -> Result<(), ValidationError> {
        let mut frame = self.get_snapshot();
        frame.time.start_bar = start_bar;
        frame.time.end_bar = end_bar;
        frame.time.fade_in_beats = fade_in_beats;
        frame.time.fade_out_beats = fade_out_beats;
        
        // Update intent ID to mark as new
        frame.meta.intent_id = self.new_intent_id();
        
        self.validate_and_store(frame)
    }
    
    /// Update provenance (convenience method)
    pub fn update_provenance(
        &self,
        source: IntentSource,
        user_override_weight: f32,
    ) -> Result<(), ValidationError> {
        let mut frame = self.get_snapshot();
        frame.provenance.source = source;
        frame.provenance.user_override_weight = user_override_weight;
        
        // Update intent ID to mark as new
        frame.meta.intent_id = self.new_intent_id();
        
        self.validate_and_store(frame)
    }
}

impl Default for IntentIRManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Global Intent IR Manager instance
use std::sync::OnceLock;

static INTENT_IR_MANAGER: OnceLock<Arc<IntentIRManager>> = OnceLock::new();

/// Get global Intent IR Manager instance
pub fn get_intent_ir_manager() -> Arc<IntentIRManager> {
    INTENT_IR_MANAGER.get_or_init(|| Arc::new(IntentIRManager::new())).clone()
}

/// Initialize Intent IR Manager with a session ID
pub fn initialize(session_id: Option<u64>) -> Arc<IntentIRManager> {
    let manager = Arc::new(IntentIRManager::new());
    
    // Set session ID if provided
    if let Some(sid) = session_id {
        let mut frame = manager.get_snapshot();
        frame.meta.session_id = sid;
        let _ = manager.validate_and_store(frame);
    } else {
        // Generate new session ID
        let sid = manager.new_session_id();
        let mut frame = manager.get_snapshot();
        frame.meta.session_id = sid;
        let _ = manager.validate_and_store(frame);
    }
    
    INTENT_IR_MANAGER.set(manager.clone()).ok();
    manager
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_manager_creation() {
        let manager = IntentIRManager::new();
        assert!(!manager.is_valid()); // Default frame has intent_id = 0
    }
    
    #[test]
    fn test_validate_and_store() {
        let manager = IntentIRManager::new();
        let frame = IntentFrame::with_ids(123, 456);
        assert!(manager.validate_and_store(frame).is_ok());
        assert!(manager.is_valid());
        assert_eq!(manager.get_current_intent_id(), 123);
    }
    
    #[test]
    fn test_get_snapshot() {
        let manager = IntentIRManager::new();
        let frame = IntentFrame::with_ids(123, 456);
        manager.validate_and_store(frame).unwrap();
        
        let snapshot = manager.get_snapshot();
        assert_eq!(snapshot.meta.intent_id, 123);
    }
    
    #[test]
    fn test_update_emotion() {
        let manager = IntentIRManager::new();
        // First create a valid frame
        let mut frame = IntentFrame::default();
        frame.meta.intent_id = manager.new_intent_id();
        frame.meta.session_id = manager.new_session_id();
        manager.validate_and_store(frame).unwrap();
        
        assert!(manager.update_emotion(0.5, 0.6, 0.7, None, 0.8, 0.9).is_ok());
        let snapshot = manager.get_snapshot();
        assert_eq!(snapshot.emotion.valence, 0.5);
        assert_eq!(snapshot.emotion.arousal, 0.6);
    }
}
