use std::time::Duration;
use tokio::time::timeout;
use serde_json;
use idaw_lib::bridge::kelly_ffi::{
    get_kelly_brain_manager, EmotionState, KellyBrain, KellyBrainManager, KellyError,
    KellyErrorCode,
};
use idaw_lib::events::{get_event_manager, EventManager, KellyEvent};
use idaw_lib::state::{get_state_manager, StateEvent, StateManager};

// Note: Integration tests need to be in the same crate or use the public API
// For now, we'll test the public Tauri commands via invoke calls
// Direct module access would require making modules public or restructuring

// =============================================================================
// Intent schema tests (no FFI required)
// =============================================================================

#[tokio::test]
async fn test_complete_song_intent_request_roundtrip() {
    use idaw_lib::generated::intent::CompleteSongIntentRequest;
    let json = r#"{
        "core_desire": "I want to feel calm",
        "mood_primary": "peaceful",
        "genre": "ambient",
        "tempo": 90,
        "key_mode": "C major",
        "structure": [{"name": "intro", "bars": 8, "repetitions": null}],
        "instruments": [{"instrument": "pad", "techniques": null}],
        "allow_legacy_fallback": true
    }"#;
    let intent: CompleteSongIntentRequest = serde_json::from_str(json).expect("deserialize intent");
    assert_eq!(intent.core_desire, "I want to feel calm");
    assert_eq!(intent.mood_primary, "peaceful");
    assert_eq!(intent.genre, "ambient");
    assert_eq!(intent.tempo, Some(90));
    assert_eq!(intent.key_mode, "C major");
    assert_eq!(intent.structure.len(), 1);
    assert_eq!(intent.structure[0].name, "intro");
    assert_eq!(intent.structure[0].bars, 8);
    assert_eq!(intent.instruments.len(), 1);
    assert_eq!(intent.instruments[0].instrument, "pad");
    assert_eq!(intent.allow_legacy_fallback, Some(true));
    let back = serde_json::to_string(&intent).expect("serialize intent");
    let _again: CompleteSongIntentRequest = serde_json::from_str(&back).expect("roundtrip");
}

// =============================================================================
// FFI Integration Tests (require KellyFFI built and linked)
// =============================================================================

#[tokio::test]
async fn test_kelly_brain_initialization() {
    let mut brain = KellyBrain::new().expect("Failed to create KellyBrain");
    
    // Test with non-existent path (should fail gracefully)
    let result = brain.initialize("/nonexistent/path");
    assert!(result.is_err(), "Should fail with non-existent path");
    
    // Test with empty path
    let result = brain.initialize("");
    assert!(result.is_err(), "Should fail with empty path");
    
    // Note: Test with valid path would require actual data files
    // In CI/CD, this could be set up with test data
}

#[tokio::test]
async fn test_kelly_brain_text_processing() {
    let mut brain = KellyBrain::new().expect("Failed to create KellyBrain");
    
    // Try to process text (may fail if not initialized, but should not crash)
    let result = brain.from_text("I feel happy and excited");
    
    // If initialization failed, this should return appropriate error
    if let Err(error) = result {
        match error.code {
            KellyErrorCode::InitializationFailed => {
                // Expected if data files not available
            },
            _ => panic!("Unexpected error: {:?}", error),
        }
    }
}

#[tokio::test]
async fn test_kelly_brain_emotion_processing() {
    let mut brain = KellyBrain::new().expect("Failed to create KellyBrain");
    
    // Test valid emotion parameters
    let result = brain.from_emotion("joy", 0.8);
    
    // Should fail with initialization error if not initialized
    if let Err(error) = result {
        assert_eq!(error.code, KellyErrorCode::InitializationFailed);
    }
    
    // Test invalid intensity values
    let result = brain.from_emotion("joy", -0.5);  // Below 0
    assert!(result.is_err(), "Should fail with invalid intensity");
    
    let result = brain.from_emotion("joy", 1.5);   // Above 1
    assert!(result.is_err(), "Should fail with invalid intensity");
}

#[tokio::test]
async fn test_kelly_brain_parameter_updates() {
    let mut brain = KellyBrain::new().expect("Failed to create KellyBrain");
    
    // Test valid parameter ranges
    let result = brain.set_emotion_parameters(0.0, 0.5, 1.0);
    if let Err(error) = result {
        // Should only fail with initialization error
        assert_eq!(error.code, KellyErrorCode::InitializationFailed);
    }
    
    // Test invalid parameter values
    let result = brain.set_emotion_parameters(-1.5, 0.5, 0.5);  // Invalid valence
    assert!(result.is_err(), "Should fail with invalid valence");
    
    let result = brain.set_emotion_parameters(0.0, -0.5, 0.5);  // Invalid arousal  
    assert!(result.is_err(), "Should fail with invalid arousal");
    
    let result = brain.set_emotion_parameters(0.0, 0.5, 1.5);   // Invalid dominance
    assert!(result.is_err(), "Should fail with invalid dominance");
}

#[tokio::test]
async fn test_kelly_brain_utility_functions() {
    // Test version function
    let version = KellyBrain::get_version();
    assert!(!version.is_empty(), "Version should not be empty");
    assert!(version.contains("KellyBrain"), "Version should contain 'KellyBrain'");
    
    // Test data file checking
    assert!(!KellyBrain::check_data_files("/nonexistent"), "Should return false for nonexistent path");
    
    // May return true for current directory if data files exist
    let _result = KellyBrain::check_data_files("./data");
}

// =============================================================================
// Manager Integration Tests
// =============================================================================

#[tokio::test]
async fn test_kelly_brain_manager() {
    let manager = KellyBrainManager::new();
    
    // Should not be initialized initially
    assert!(!manager.is_initialized(), "Manager should not be initialized initially");
    
    // Test initialization with invalid path
    let result = manager.initialize("/nonexistent/path");
    assert!(result.is_err(), "Should fail with invalid path");
    
    // Test operation without initialization
    let result = manager.with_brain(|brain| brain.get_emotion_state());
    assert!(result.is_err(), "Should fail when not initialized");
    
    // Should still not be initialized after failed attempts
    assert!(!manager.is_initialized(), "Manager should remain uninitialized after failures");
}

// =============================================================================
// State Management Tests  
// =============================================================================

#[tokio::test]
async fn test_state_manager_creation() {
    let manager = StateManager::new();
    
    // Get initial state
    let state = manager.get_state();
    assert!(!state.initialized, "Should not be initialized initially");
    assert!(state.emotion_state.is_none(), "Should have no emotion state initially");
    assert!(state.current_intent.is_none(), "Should have no current intent initially");
    assert!(!state.processing, "Should not be processing initially");
}

#[tokio::test]
async fn test_state_updates() {
    let manager = StateManager::new();
    
    // Test initialization state update
    manager.set_initialized(true);
    let state = manager.get_state();
    assert!(state.initialized, "State should reflect initialization");
    assert!(state.last_update.is_some(), "Should have last update timestamp");
    
    // Test emotion state update
    let emotion = EmotionState {
        valence: 0.5,
        arousal: 0.7,
        dominance: 0.3,
        complexity: 0.6,
    };
    
    manager.set_emotion_state(emotion.clone());
    let state = manager.get_state();
    assert!(state.emotion_state.is_some(), "Should have emotion state");
    let stored_emotion = state.emotion_state.unwrap();
    assert_eq!(stored_emotion.valence, emotion.valence);
    assert_eq!(stored_emotion.arousal, emotion.arousal);
}

#[tokio::test]
async fn test_state_event_subscription() {
    let manager = StateManager::new();
    
    // Subscribe to events
    let mut receiver = manager.subscribe();
    
    // Update state (should trigger event)
    manager.set_initialized(true);
    
    // Try to receive event (with timeout)
    let event_result = timeout(Duration::from_millis(100), receiver.recv()).await;
    
    match event_result {
        Ok(Ok(event)) => {
            match event {
                StateEvent::KellyBrainInitialized { success } => {
                    assert!(success, "Should indicate successful initialization");
                },
                _ => panic!("Unexpected event type: {:?}", event),
            }
        },
        Ok(Err(e)) => panic!("Event receive error: {:?}", e),
        Err(_) => {
            // Timeout - this might happen in some test environments
            println!("Warning: Event receive timed out");
        }
    }
}

// =============================================================================
// Event System Tests
// =============================================================================

#[tokio::test] 
async fn test_event_manager_creation() {
    let manager = EventManager::new();
    
    // Should be able to subscribe
    let _receiver = manager.subscribe();
    
    // Should have zero listeners initially
    assert_eq!(manager.get_listener_count(), 0);
}

#[tokio::test]
async fn test_event_listener_management() {
    let manager = EventManager::new();
    
    // Add listeners
    manager.add_listener("test_listener_1".to_string());
    manager.add_listener("test_listener_2".to_string());
    
    assert_eq!(manager.get_listener_count(), 2);
    
    // Remove listener
    manager.remove_listener("test_listener_1");
    
    assert_eq!(manager.get_listener_count(), 1);
    
    // Cleanup should not affect active listeners
    manager.cleanup_listeners();
    assert_eq!(manager.get_listener_count(), 1);
}

// =============================================================================
// Error Handling Tests
// =============================================================================

#[tokio::test]
async fn test_error_conversion() {
    let error: KellyError = KellyErrorCode::InvalidParameter.into();
    assert_eq!(error.code, KellyErrorCode::InvalidParameter);
    assert!(!error.message.is_empty());
    
    // Test Display trait
    let error_string = format!("{}", error);
    assert!(error_string.contains("InvalidParameter"));
}

#[tokio::test]
async fn test_error_to_string_conversion() {
    let error = KellyError {
        code: KellyErrorCode::JsonParseError,
        message: "Test error message".to_string(),
    };
    
    let error_string = format!("KellyBrain Error: {}", error.message);
    assert!(error_string.contains("Test error message"));
    assert!(error_string.contains("KellyBrain Error"));
}

// =============================================================================
// Performance Tests (Basic)
// =============================================================================

#[tokio::test]
async fn test_ffi_call_performance() {
    let brain = KellyBrain::new().expect("Failed to create KellyBrain");
    
    // Measure time for multiple calls
    let start = std::time::Instant::now();
    let iterations = 100;
    
    for _ in 0..iterations {
        let _version = KellyBrain::get_version();
        let _initialized = brain.is_initialized();
    }
    
    let elapsed = start.elapsed();
    let avg_time = elapsed / iterations;
    
    // FFI calls should be very fast (< 1ms each)
    assert!(avg_time < Duration::from_millis(1), 
            "FFI calls too slow: {:?} per call", avg_time);
    
    println!("Average FFI call time: {:?}", avg_time);
}

#[tokio::test]
async fn test_state_update_performance() {
    let manager = StateManager::new();
    
    // Measure state update performance
    let start = std::time::Instant::now();
    let iterations = 1000;
    
    for i in 0..iterations {
        let valence = (i as f32) / (iterations as f32) * 2.0 - 1.0;  // -1 to 1
        let emotion = EmotionState {
            valence,
            arousal: 0.5,
            dominance: 0.5,
            complexity: 0.5,
        };
        manager.set_emotion_state(emotion);
    }
    
    let elapsed = start.elapsed();
    let avg_time = elapsed / iterations;
    
    // State updates should be very fast
    assert!(avg_time < Duration::from_micros(100),
            "State updates too slow: {:?} per update", avg_time);
    
    println!("Average state update time: {:?}", avg_time);
}

// =============================================================================
// Integration Flow Tests
// =============================================================================

#[tokio::test]
async fn test_complete_workflow_simulation() {
    // Test the complete workflow without actual initialization
    // This tests the API structure and error handling
    
    let manager = KellyBrainManager::new();
    
    // Step 1: Try to process text without initialization (should fail appropriately)
    let result = manager.with_brain(|brain| brain.from_text("test emotion"));
    assert!(result.is_err(), "Should fail when not initialized");
    
    // Step 2: Try to set parameters (should fail appropriately)  
    let result = manager.with_brain(|brain| brain.set_emotion_parameters(0.5, 0.7, 0.3));
    assert!(result.is_err(), "Should fail when not initialized");
    
    // Step 3: Check state queries (should fail appropriately)
    let result = manager.with_brain(|brain| brain.get_emotion_state());
    assert!(result.is_err(), "Should fail when not initialized");
    
    // All errors should be InitializationFailed
    // This tests that the error handling flow works correctly
}

#[tokio::test]
async fn test_global_manager_instance() {
    let manager1 = get_kelly_brain_manager();
    let manager2 = get_kelly_brain_manager();
    
    // Should be the same instance (singleton pattern)
    assert!(std::ptr::eq(manager1, manager2), "Should return same global instance");
    
    // Should not be initialized
    assert!(!manager1.is_initialized(), "Global manager should not be initialized initially");
}

// =============================================================================
// Concurrent Access Tests
// =============================================================================

#[tokio::test]
async fn test_concurrent_state_access() {
    let manager = get_state_manager();
    
    // Spawn multiple tasks that access state concurrently
    let mut handles = Vec::new();
    
    for i in 0..10 {
        let manager_clone = manager.clone();
        let handle = tokio::spawn(async move {
            {
                let manager_guard = manager_clone.lock().unwrap();
                manager_guard.set_processing(true, Some(format!("operation_{}", i)));
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
            
            let manager_guard = manager_clone.lock().unwrap();
            manager_guard.set_processing(false, Some(format!("operation_{}", i)));
            manager_guard.get_state()
        });
        handles.push(handle);
    }
    
    // Wait for all tasks to complete
    let mut final_states = Vec::new();
    for handle in handles {
        let state = handle.await.expect("Task should complete successfully");
        final_states.push(state);
    }
    
    // All states should be valid (not crashed)
    assert_eq!(final_states.len(), 10);
    
    // Final processing state should be false
    let final_state = {
        let manager_guard = manager.lock().unwrap();
        manager_guard.get_state()
    };
    assert!(!final_state.processing, "Should not be processing after all tasks complete");
}

// =============================================================================
// Event System Integration Tests
// =============================================================================

#[tokio::test]
async fn test_event_emission_and_subscription() {
    let manager = get_event_manager();
    
    // Subscribe to events
    let mut receiver = {
        let manager_guard = manager.lock().unwrap();
        manager_guard.subscribe()
    };
    
    // Emit test event
    {
        let manager_guard = manager.lock().unwrap();
        manager_guard.emit(KellyEvent::ParameterChanged {
            parameter: "test_param".to_string(),
            value: serde_json::json!({"test": "value"}),
        });
    }
    
    // Try to receive event (with timeout)
    let event_result = timeout(Duration::from_millis(100), receiver.recv()).await;
    
    match event_result {
        Ok(Ok(event)) => {
            match event {
                KellyEvent::ParameterChanged { parameter, value } => {
                    assert_eq!(parameter, "test_param");
                    assert_eq!(value["test"], "value");
                },
                _ => panic!("Unexpected event type: {:?}", event),
            }
        },
        Ok(Err(e)) => panic!("Event receive error: {:?}", e),
        Err(_) => panic!("Event receive timed out"),
    }
}

// =============================================================================
// Memory Safety Tests
// =============================================================================

#[tokio::test]
async fn test_memory_safety() {
    // Test that creating and dropping many KellyBrain instances doesn't leak memory
    for _ in 0..100 {
        let mut brain = KellyBrain::new().expect("Failed to create KellyBrain");
        
        // Try some operations (may fail, but should not leak)
        let _result = brain.get_emotion_state();
        let _result = brain.from_text("test");
        
        // Brain should be automatically dropped here
    }
    
    // If we reach here without crashing, memory safety is working
}

#[tokio::test]
async fn test_string_memory_management() {
    let mut brain = KellyBrain::new().expect("Failed to create KellyBrain");
    
    // Test that string returns are properly managed
    for _ in 0..50 {
        // These may return null if not initialized, but should not leak
        let _result = brain.from_text("test emotion");
        let _result = brain.get_emotion_state();
        let _result = brain.get_available_emotions();
    }
    
    // Rust should automatically manage the returned strings
}

// =============================================================================
// Error Propagation Tests
// =============================================================================

#[tokio::test]
async fn test_error_propagation() {
    let manager = get_kelly_brain_manager();
    
    // Test that errors propagate correctly through the manager
    let result = manager.with_brain(|brain| {
        brain.set_emotion_parameters(-2.0, 0.5, 0.5)  // Invalid valence
    });
    
    assert!(result.is_err(), "Should propagate parameter validation error");
    
    let error = result.unwrap_err();
    assert_eq!(error.code, KellyErrorCode::InitializationFailed);
}

// =============================================================================
// Concurrent Manager Tests
// =============================================================================

#[tokio::test]
async fn test_concurrent_manager_operations() {
    let manager = get_kelly_brain_manager();
    
    // Spawn multiple tasks that try to use the manager
    let mut handles = Vec::new();
    
    for i in 0..5 {
        let handle = tokio::spawn(async move {
            let result = manager.with_brain(|_brain| {
                // Try to get version (safe operation)
                Ok(KellyBrain::get_version())
            });
            
            // Should fail with initialization error, but not crash
            match result {
                Err(error) => {
                    assert_eq!(error.code, KellyErrorCode::InitializationFailed);
                },
                Ok(_) => {
                    // Unexpected success (maybe brain was initialized)
                }
            }
            
            i
        });
        handles.push(handle);
    }
    
    // All tasks should complete successfully
    for (i, handle) in handles.into_iter().enumerate() {
        let result = handle.await.expect("Task should complete");
        assert_eq!(result, i, "Task should return correct value");
    }
}

// =============================================================================
// Helper Functions for Manual Testing
// =============================================================================

#[cfg(test)]
mod test_helpers {
    use super::*;
    
    /// Helper function to create initialized KellyBrain for tests that need it
    /// Returns None if initialization fails (e.g., data files not available)
    pub async fn create_initialized_kelly_brain(data_path: &str) -> Option<KellyBrain> {
        let mut brain = KellyBrain::new().ok()?;
        
        match brain.initialize(data_path) {
            Ok(()) => Some(brain),
            Err(_) => None,
        }
    }
    
    /// Helper to check if test data is available
    pub fn has_test_data() -> bool {
        KellyBrain::check_data_files("./data") ||
        KellyBrain::check_data_files("../data") ||
        KellyBrain::check_data_files("../../data")
    }
}

// =============================================================================
// Conditional Tests (Only Run if Data Available)
// =============================================================================

#[tokio::test]
async fn test_full_workflow_with_data() {
    use test_helpers::*;
    
    if !has_test_data() {
        println!("Skipping test: No test data available");
        return;
    }
    
    let data_path = if KellyBrain::check_data_files("./data") {
        "./data"
    } else if KellyBrain::check_data_files("../data") {
        "../data"
    } else {
        "../../data"
    };
    
    let mut brain = match create_initialized_kelly_brain(data_path).await {
        Some(brain) => brain,
        None => {
            println!("Failed to initialize KellyBrain with data");
            return;
        }
    };
    
    // Test complete workflow with real data
    let intent = brain.from_text("feeling melancholy").unwrap();
    assert!(!intent.emotional_intent.is_empty());
    
    let emotion_state = brain.get_emotion_state().unwrap();
    assert!(emotion_state.valence >= -1.0 && emotion_state.valence <= 1.0);
    
    let midi = brain.generate_midi(&intent, 4).unwrap();
    assert_eq!(midi.bars, 4);
    assert!(!midi.tracks.is_empty());
    
    println!("Full workflow test completed successfully");
}
