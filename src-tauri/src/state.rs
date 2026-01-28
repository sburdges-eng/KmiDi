use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::sync::broadcast;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager};
use crate::bridge::kelly_ffi::{EmotionState, IntentResult, GeneratedMidi};

// =============================================================================
// State Data Types
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KellyBrainState {
    pub initialized: bool,
    pub emotion_state: Option<EmotionState>,
    pub current_intent: Option<IntentResult>,
    pub current_midi: Option<GeneratedMidi>,
    pub processing: bool,
    pub last_update: Option<String>,
}

impl Default for KellyBrainState {
    fn default() -> Self {
        KellyBrainState {
            initialized: false,
            emotion_state: None,
            current_intent: None,
            current_midi: None,
            processing: false,
            last_update: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum StateEvent {
    KellyBrainInitialized { success: bool },
    EmotionStateChanged { emotion_state: EmotionState },
    IntentGenerated { intent: IntentResult },
    MidiGenerated { midi: GeneratedMidi },
    ProcessingStarted { operation: String },
    ProcessingCompleted { operation: String },
    Error { message: String },
}

// =============================================================================
// State Manager
// =============================================================================

pub struct StateManager {
    state: Arc<Mutex<KellyBrainState>>,
    event_sender: broadcast::Sender<StateEvent>,
    app_handle: Option<AppHandle>,
    subscribers: Arc<Mutex<HashMap<String, Instant>>>,
}

impl StateManager {
    pub fn new() -> Self {
        let (event_sender, _) = broadcast::channel(100);
        
        StateManager {
            state: Arc::new(Mutex::new(KellyBrainState::default())),
            event_sender,
            app_handle: None,
            subscribers: Arc::new(Mutex::new(HashMap::new())),
        }
    }
    
    /// Set Tauri app handle for event emission
    pub fn set_app_handle(&mut self, handle: AppHandle) {
        self.app_handle = Some(handle);
    }
    
    /// Get current state snapshot
    pub fn get_state(&self) -> KellyBrainState {
        let state_guard = self.state.lock().unwrap();
        state_guard.clone()
    }
    
    /// Update KellyBrain initialization status
    pub fn set_initialized(&self, initialized: bool) {
        {
            let mut state_guard = self.state.lock().unwrap();
            state_guard.initialized = initialized;
            state_guard.last_update = Some(chrono::Utc::now().to_rfc3339());
        }
        
        let event = StateEvent::KellyBrainInitialized { success: initialized };
        self.emit_event(event);
    }
    
    /// Update emotion state
    pub fn set_emotion_state(&self, emotion_state: EmotionState) {
        {
            let mut state_guard = self.state.lock().unwrap();
            state_guard.emotion_state = Some(emotion_state.clone());
            state_guard.last_update = Some(chrono::Utc::now().to_rfc3339());
        }
        
        let event = StateEvent::EmotionStateChanged { emotion_state };
        self.emit_event(event);
    }
    
    /// Update current intent
    pub fn set_current_intent(&self, intent: IntentResult) {
        {
            let mut state_guard = self.state.lock().unwrap();
            state_guard.current_intent = Some(intent.clone());
            state_guard.last_update = Some(chrono::Utc::now().to_rfc3339());
        }
        
        let event = StateEvent::IntentGenerated { intent };
        self.emit_event(event);
    }
    
    /// Update current MIDI
    pub fn set_current_midi(&self, midi: GeneratedMidi) {
        {
            let mut state_guard = self.state.lock().unwrap();
            state_guard.current_midi = Some(midi.clone());
            state_guard.last_update = Some(chrono::Utc::now().to_rfc3339());
        }
        
        let event = StateEvent::MidiGenerated { midi };
        self.emit_event(event);
    }
    
    /// Set processing status
    pub fn set_processing(&self, processing: bool, operation: Option<String>) {
        {
            let mut state_guard = self.state.lock().unwrap();
            state_guard.processing = processing;
            state_guard.last_update = Some(chrono::Utc::now().to_rfc3339());
        }
        
        if let Some(op) = operation {
            let event = if processing {
                StateEvent::ProcessingStarted { operation: op }
            } else {
                StateEvent::ProcessingCompleted { operation: op }
            };
            self.emit_event(event);
        }
    }
    
    /// Emit error event
    pub fn emit_error(&self, message: String) {
        let event = StateEvent::Error { message };
        self.emit_event(event);
    }
    
    /// Subscribe to state events
    pub fn subscribe(&self) -> broadcast::Receiver<StateEvent> {
        self.event_sender.subscribe()
    }
    
    /// Add subscriber tracking
    pub fn add_subscriber(&self, id: String) {
        let mut subscribers = self.subscribers.lock().unwrap();
        subscribers.insert(id, Instant::now());
    }
    
    /// Remove subscriber tracking
    pub fn remove_subscriber(&self, id: &str) {
        let mut subscribers = self.subscribers.lock().unwrap();
        subscribers.remove(id);
    }
    
    /// Get subscriber count
    pub fn get_subscriber_count(&self) -> usize {
        let subscribers = self.subscribers.lock().unwrap();
        subscribers.len()
    }
    
    /// Cleanup old subscribers (inactive for > 30 minutes)
    pub fn cleanup_subscribers(&self) {
        let mut subscribers = self.subscribers.lock().unwrap();
        let now = Instant::now();
        subscribers.retain(|_, &mut last_seen| {
            now.duration_since(last_seen) < Duration::from_secs(1800) // 30 minutes
        });
    }
    
    /// Emit event to all subscribers
    fn emit_event(&self, event: StateEvent) {
        // Send to broadcast channel
        let _ = self.event_sender.send(event.clone());
        
        // Send to Tauri frontend if available
        if let Some(ref app_handle) = self.app_handle {
            let _ = app_handle.emit("kelly-brain-state", &event);
        }
    }
}

// =============================================================================
// Global State Manager Instance
// =============================================================================

use std::sync::OnceLock;

static STATE_MANAGER: OnceLock<Arc<Mutex<StateManager>>> = OnceLock::new();

/// Get global state manager instance
pub fn get_state_manager() -> Arc<Mutex<StateManager>> {
    STATE_MANAGER.get_or_init(|| Arc::new(Mutex::new(StateManager::new()))).clone()
}

/// Initialize state manager with Tauri app handle
pub fn initialize_state_manager(app_handle: AppHandle) {
    let manager = get_state_manager();
    let mut manager_guard = manager.lock().unwrap();
    manager_guard.set_app_handle(app_handle);
}

// =============================================================================
// Tauri Commands for State Management
// =============================================================================

#[tauri::command]
pub async fn get_kelly_brain_state() -> Result<KellyBrainState, String> {
    let manager = get_state_manager();
    let manager_guard = manager.lock().unwrap();
    Ok(manager_guard.get_state())
}

#[tauri::command]
pub async fn subscribe_to_state_events(subscriber_id: String) -> Result<bool, String> {
    let manager = get_state_manager();
    let manager_guard = manager.lock().unwrap();
    manager_guard.add_subscriber(subscriber_id);
    Ok(true)
}

#[tauri::command]
pub async fn unsubscribe_from_state_events(subscriber_id: String) -> Result<bool, String> {
    let manager = get_state_manager();
    let manager_guard = manager.lock().unwrap();
    manager_guard.remove_subscriber(&subscriber_id);
    Ok(true)
}

#[tauri::command]
pub async fn get_subscriber_count() -> Result<usize, String> {
    let manager = get_state_manager();
    let manager_guard = manager.lock().unwrap();
    Ok(manager_guard.get_subscriber_count())
}

// =============================================================================
// Background Tasks
// =============================================================================

/// Start background tasks for state management
pub fn start_background_tasks() {
    // Cleanup task
    tokio::spawn(async {
        let mut interval = tokio::time::interval(Duration::from_secs(300)); // 5 minutes
        
        loop {
            interval.tick().await;
            
            let manager = get_state_manager();
            let manager_guard = manager.lock().unwrap();
            manager_guard.cleanup_subscribers();
        }
    });
    
    // State sync task - uses direct FFI calls to avoid circular dependency
    tokio::spawn(async {
        let mut interval = tokio::time::interval(Duration::from_secs(10)); // 10 seconds
        
        loop {
            interval.tick().await;
            
            // Use FFI directly, not Tauri commands, to avoid circular dependency
            let manager = crate::bridge::kelly_ffi::get_kelly_brain_manager();
            
            if manager.is_initialized() {
                // Sync emotion state from C++ backend
                let emotion_result = manager.with_brain(|brain| {
                    brain.get_emotion_state()
                });
                
                if let Ok(emotion_state) = emotion_result {
                    let state_manager = get_state_manager();
                    let mut state_guard = state_manager.lock().unwrap();
                    state_guard.set_emotion_state(emotion_state.valence, emotion_state.arousal, emotion_state.dominance, emotion_state.complexity);
                }
            }
        }
    });
}

// =============================================================================
// Helper Functions for Commands
// =============================================================================

/// Update state when KellyBrain operations complete
pub fn update_state_after_operation<T>(
    operation: &str, 
    result: &Result<T, String>
) where 
    T: Clone,
{
    let manager = get_state_manager();
    let manager_guard = manager.lock().unwrap();
    
    manager_guard.set_processing(false, Some(operation.to_string()));
    
    if let Err(ref error) = result {
        manager_guard.emit_error(error.clone());
    }
}

/// Set processing state before operation
pub fn set_processing_state(operation: &str) {
    let manager = get_state_manager();
    let manager_guard = manager.lock().unwrap();
    manager_guard.set_processing(true, Some(operation.to_string()));
}

/// Update state with intent result
pub fn update_state_with_intent(intent: &IntentResult) {
    let manager = get_state_manager();
    let manager_guard = manager.lock().unwrap();
    manager_guard.set_current_intent(intent.clone());
}

/// Update state with MIDI result
pub fn update_state_with_midi(midi: &GeneratedMidi) {
    let manager = get_state_manager();
    let manager_guard = manager.lock().unwrap();
    manager_guard.set_current_midi(midi.clone());
}

/// Update initialization state
pub fn update_initialization_state(initialized: bool) {
    let manager = get_state_manager();
    let manager_guard = manager.lock().unwrap();
    manager_guard.set_initialized(initialized);
}