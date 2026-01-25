use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager, Emitter};
use tokio::sync::broadcast;

use crate::state::{StateEvent, get_state_manager, KellyBrainState};

// =============================================================================
// Event Types
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum KellyEvent {
    // State events (forwarded from StateManager)
    StateChanged { state: crate::state::KellyBrainState },
    EmotionUpdate { valence: f32, arousal: f32, dominance: f32, complexity: f32 },
    IntentGenerated { intent: crate::bridge::kelly_ffi::IntentResult },
    MidiGenerated { midi: crate::bridge::kelly_ffi::GeneratedMidi },
    
    // Processing events
    ProcessingStarted { operation: String, timestamp: String },
    ProcessingProgress { operation: String, progress: f32 },
    ProcessingCompleted { operation: String, timestamp: String },
    
    // System events
    KellyBrainInitialized { success: bool, version: Option<String> },
    ConnectionStatusChanged { connected: bool },
    Error { message: String, timestamp: String },
    
    // Real-time updates
    ParameterChanged { parameter: String, value: serde_json::Value },
    AudioAnalysisUpdate { analysis: serde_json::Value },
}

// =============================================================================
// Event Manager
// =============================================================================

pub struct EventManager {
    app_handle: Option<AppHandle>,
    event_sender: broadcast::Sender<KellyEvent>,
    listeners: Arc<Mutex<HashMap<String, Instant>>>,
    state_receiver: Option<broadcast::Receiver<StateEvent>>,
}

impl EventManager {
    pub fn new() -> Self {
        let (event_sender, _) = broadcast::channel(200);
        
        EventManager {
            app_handle: None,
            event_sender,
            listeners: Arc::new(Mutex::new(HashMap::new())),
            state_receiver: None,
        }
    }
    
    /// Initialize with Tauri app handle
    pub fn initialize(&mut self, app_handle: AppHandle) {
        self.app_handle = Some(app_handle);
        
        // Subscribe to state events
        let state_manager = get_state_manager();
        let state_manager_guard = state_manager.lock().unwrap();
        self.state_receiver = Some(state_manager_guard.subscribe());
        drop(state_manager_guard);
        
        // Start event forwarding task
        self.start_event_forwarding();
    }
    
    /// Emit event to all listeners
    pub fn emit(&self, event: KellyEvent) {
        // Send to broadcast channel
        let _ = self.event_sender.send(event.clone());
        
        // Send to Tauri frontend
        if let Some(ref app_handle) = self.app_handle {
            let event_name = match &event {
                KellyEvent::StateChanged { .. } => "kelly-state-changed",
                KellyEvent::EmotionUpdate { .. } => "kelly-emotion-update",
                KellyEvent::IntentGenerated { .. } => "kelly-intent-generated",
                KellyEvent::MidiGenerated { .. } => "kelly-midi-generated",
                KellyEvent::ProcessingStarted { .. } => "kelly-processing-started",
                KellyEvent::ProcessingProgress { .. } => "kelly-processing-progress",
                KellyEvent::ProcessingCompleted { .. } => "kelly-processing-completed",
                KellyEvent::KellyBrainInitialized { .. } => "kelly-brain-initialized",
                KellyEvent::ConnectionStatusChanged { .. } => "kelly-connection-status",
                KellyEvent::Error { .. } => "kelly-error",
                KellyEvent::ParameterChanged { .. } => "kelly-parameter-changed",
                KellyEvent::AudioAnalysisUpdate { .. } => "kelly-audio-analysis",
            };
            
            let _ = app_handle.emit(event_name, &event);
        }
    }
    
    /// Subscribe to events
    pub fn subscribe(&self) -> broadcast::Receiver<KellyEvent> {
        self.event_sender.subscribe()
    }
    
    /// Add event listener
    pub fn add_listener(&self, listener_id: String) {
        let mut listeners = self.listeners.lock().unwrap();
        listeners.insert(listener_id, Instant::now());
    }
    
    /// Remove event listener
    pub fn remove_listener(&self, listener_id: &str) {
        let mut listeners = self.listeners.lock().unwrap();
        listeners.remove(listener_id);
    }
    
    /// Get listener count
    pub fn get_listener_count(&self) -> usize {
        let listeners = self.listeners.lock().unwrap();
        listeners.len()
    }
    
    /// Cleanup inactive listeners
    pub fn cleanup_listeners(&self) {
        let mut listeners = self.listeners.lock().unwrap();
        let now = Instant::now();
        listeners.retain(|_, &mut last_seen| {
            now.duration_since(last_seen) < Duration::from_secs(1800) // 30 minutes
        });
    }
    
    /// Start event forwarding from state manager
    fn start_event_forwarding(&mut self) {
        if let Some(mut receiver) = self.state_receiver.take() {
            let event_sender = self.event_sender.clone();
            let app_handle = self.app_handle.clone();
            
            tokio::spawn(async move {
                loop {
                    match receiver.recv().await {
                        Ok(state_event) => {
                            // Convert StateEvent to KellyEvent and forward
                            let kelly_event = match state_event {
                                StateEvent::KellyBrainInitialized { success } => {
                                    let version = if success {
                                        Some(crate::bridge::kelly_ffi::KellyBrain::get_version())
                                    } else {
                                        None
                                    };
                                    KellyEvent::KellyBrainInitialized { success, version }
                                },
                                StateEvent::EmotionStateChanged { emotion_state } => {
                                    KellyEvent::EmotionUpdate {
                                        valence: emotion_state.valence,
                                        arousal: emotion_state.arousal,
                                        dominance: emotion_state.dominance,
                                        complexity: emotion_state.complexity,
                                    }
                                },
                                StateEvent::IntentGenerated { intent } => {
                                    KellyEvent::IntentGenerated { intent }
                                },
                                StateEvent::MidiGenerated { midi } => {
                                    KellyEvent::MidiGenerated { midi }
                                },
                                StateEvent::ProcessingStarted { operation } => {
                                    KellyEvent::ProcessingStarted {
                                        operation,
                                        timestamp: chrono::Utc::now().to_rfc3339(),
                                    }
                                },
                                StateEvent::ProcessingCompleted { operation } => {
                                    KellyEvent::ProcessingCompleted {
                                        operation,
                                        timestamp: chrono::Utc::now().to_rfc3339(),
                                    }
                                },
                                StateEvent::Error { message } => {
                                    KellyEvent::Error {
                                        message,
                                        timestamp: chrono::Utc::now().to_rfc3339(),
                                    }
                                },
                            };
                            
                            // Forward to broadcast channel
                            let _ = event_sender.send(kelly_event.clone());
                            
                            // Forward to Tauri frontend
                            if let Some(ref app) = app_handle {
                                let event_name = match &kelly_event {
                                    KellyEvent::KellyBrainInitialized { .. } => "kelly-brain-initialized",
                                    KellyEvent::EmotionUpdate { .. } => "kelly-emotion-update",
                                    KellyEvent::IntentGenerated { .. } => "kelly-intent-generated",
                                    KellyEvent::MidiGenerated { .. } => "kelly-midi-generated",
                                    KellyEvent::ProcessingStarted { .. } => "kelly-processing-started",
                                    KellyEvent::ProcessingCompleted { .. } => "kelly-processing-completed",
                                    KellyEvent::Error { .. } => "kelly-error",
                                    _ => "kelly-event",
                                };
                                
                                let _ = app.emit(event_name, &kelly_event);
                            }
                        },
                        Err(broadcast::error::RecvError::Lagged(missed)) => {
                            eprintln!("Event forwarding lagged: missed {} events", missed);
                        },
                        Err(broadcast::error::RecvError::Closed) => {
                            eprintln!("State event channel closed");
                            break;
                        },
                    }
                }
            });
        }
    }
}

// =============================================================================
// Global Event Manager
// =============================================================================

use std::sync::OnceLock;

static EVENT_MANAGER: OnceLock<Arc<Mutex<EventManager>>> = OnceLock::new();

/// Get global event manager instance
pub fn get_event_manager() -> Arc<Mutex<EventManager>> {
    EVENT_MANAGER.get_or_init(|| Arc::new(Mutex::new(EventManager::new()))).clone()
}

/// Initialize event manager with Tauri app handle
pub fn initialize_event_manager(app_handle: AppHandle) {
    let manager = get_event_manager();
    let mut manager_guard = manager.lock().unwrap();
    manager_guard.initialize(app_handle);
}

/// Start background event system tasks
pub fn start_event_tasks() {
    // Cleanup task for inactive listeners
    tokio::spawn(async {
        let mut interval = tokio::time::interval(Duration::from_secs(300)); // 5 minutes
        
        loop {
            interval.tick().await;
            
            let manager = get_event_manager();
            let manager_guard = manager.lock().unwrap();
            manager_guard.cleanup_listeners();
        }
    });
    
    // Connection monitoring task - uses direct FFI calls to avoid circular dependency
    tokio::spawn(async {
        let mut interval = tokio::time::interval(Duration::from_secs(30)); // 30 seconds
        let mut last_connection_status = false;
        
        loop {
            interval.tick().await;
            
            // Check KellyBrain connection status via FFI directly
            let manager = crate::bridge::kelly_ffi::get_kelly_brain_manager();
            let is_connected = manager.is_initialized();
            
            // Emit connection status change event if it changed
            if is_connected != last_connection_status {
                last_connection_status = is_connected;
                
                let event_manager = get_event_manager();
                let event_guard = event_manager.lock().unwrap();
                
                let event = KellyEvent::ConnectionStatusChanged { 
                    connected: is_connected 
                };
                
                event_guard.emit(event);
            }
        }
    });
}

// =============================================================================
// Tauri Commands for Event Management
// =============================================================================

#[tauri::command]
pub async fn add_event_listener(listener_id: String) -> Result<bool, String> {
    let manager = get_event_manager();
    let manager_guard = manager.lock().unwrap();
    manager_guard.add_listener(listener_id);
    Ok(true)
}

#[tauri::command]
pub async fn remove_event_listener(listener_id: String) -> Result<bool, String> {
    let manager = get_event_manager();
    let manager_guard = manager.lock().unwrap();
    manager_guard.remove_listener(&listener_id);
    Ok(true)
}

#[tauri::command]
pub async fn get_event_listener_count() -> Result<usize, String> {
    let manager = get_event_manager();
    let manager_guard = manager.lock().unwrap();
    Ok(manager_guard.get_listener_count())
}

#[tauri::command]
pub async fn emit_test_event() -> Result<bool, String> {
    let manager = get_event_manager();
    let manager_guard = manager.lock().unwrap();
    
    manager_guard.emit(KellyEvent::ParameterChanged {
        parameter: "test_parameter".to_string(),
        value: serde_json::json!({"test": "value"}),
    });
    
    Ok(true)
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Emit emotion update event
pub fn emit_emotion_update(valence: f32, arousal: f32, dominance: f32, complexity: f32) {
    let manager = get_event_manager();
    let manager_guard = manager.lock().unwrap();
    manager_guard.emit(KellyEvent::EmotionUpdate {
        valence,
        arousal,
        dominance,
        complexity,
    });
}

/// Emit processing progress event
pub fn emit_processing_progress(operation: &str, progress: f32) {
    let manager = get_event_manager();
    let manager_guard = manager.lock().unwrap();
    manager_guard.emit(KellyEvent::ProcessingProgress {
        operation: operation.to_string(),
        progress,
    });
}

/// Emit parameter changed event
pub fn emit_parameter_changed(parameter: &str, value: serde_json::Value) {
    let manager = get_event_manager();
    let manager_guard = manager.lock().unwrap();
    manager_guard.emit(KellyEvent::ParameterChanged {
        parameter: parameter.to_string(),
        value,
    });
}

/// Emit error event
pub fn emit_error(message: &str) {
    let manager = get_event_manager();
    let manager_guard = manager.lock().unwrap();
    manager_guard.emit(KellyEvent::Error {
        message: message.to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
    });
}

/// Emit audio analysis update
pub fn emit_audio_analysis_update(analysis: serde_json::Value) {
    let manager = get_event_manager();
    let manager_guard = manager.lock().unwrap();
    manager_guard.emit(KellyEvent::AudioAnalysisUpdate { analysis });
}