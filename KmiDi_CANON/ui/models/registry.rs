//! Model Registry - Metadata, load/unload tracking, memory budget

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Model role classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelRole {
    /// Always loaded (resident)
    Resident,
    /// Loaded on demand
    OnDemand,
    /// Offline only (never loaded during audio)
    Offline,
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub name: String,
    pub size_params: u64, // Parameters in millions
    pub size_bytes: u64,   // Memory size in bytes
    pub role: ModelRole,
    pub model_type: String, // "emotion", "intent", "harmony", etc.
    pub quantized: bool,
}

/// Model load state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelLoadState {
    Unloaded,
    Loading,
    Loaded,
    Error,
}

/// Model registry entry
struct ModelEntry {
    metadata: ModelMetadata,
    state: ModelLoadState,
    load_count: u64,
}

/// Model Registry
pub struct ModelRegistry {
    models: Arc<Mutex<HashMap<String, ModelEntry>>>,
    total_memory_used: Arc<Mutex<u64>>,
    memory_budget: u64,
}

impl ModelRegistry {
    pub fn new(memory_budget: u64) -> Self {
        Self {
            models: Arc::new(Mutex::new(HashMap::new())),
            total_memory_used: Arc::new(Mutex::new(0)),
            memory_budget,
        }
    }
    
    /// Register a model
    pub fn register(&self, metadata: ModelMetadata) {
        let mut models = self.models.lock().unwrap();
        models.insert(metadata.name.clone(), ModelEntry {
            metadata,
            state: ModelLoadState::Unloaded,
            load_count: 0,
        });
    }
    
    /// Get model metadata
    pub fn get_metadata(&self, name: &str) -> Option<ModelMetadata> {
        let models = self.models.lock().unwrap();
        models.get(name).map(|e| e.metadata.clone())
    }
    
    /// Check if model can be loaded (memory budget check)
    pub fn can_load(&self, name: &str) -> bool {
        let models = self.models.lock().unwrap();
        let total_used = *self.total_memory_used.lock().unwrap();
        
        if let Some(entry) = models.get(name) {
            if entry.state == ModelLoadState::Loaded {
                return true; // Already loaded
            }
            
            // Check if loading would exceed budget
            total_used + entry.metadata.size_bytes <= self.memory_budget
        } else {
            false
        }
    }
    
    /// Mark model as loading
    pub fn mark_loading(&self, name: &str) {
        let mut models = self.models.lock().unwrap();
        if let Some(entry) = models.get_mut(name) {
            entry.state = ModelLoadState::Loading;
        }
    }
    
    /// Mark model as loaded
    pub fn mark_loaded(&self, name: &str) {
        let mut models = self.models.lock().unwrap();
        let mut total_used = self.total_memory_used.lock().unwrap();
        
        if let Some(entry) = models.get_mut(name) {
            entry.state = ModelLoadState::Loaded;
            entry.load_count += 1;
            *total_used += entry.metadata.size_bytes;
        }
    }
    
    /// Mark model as unloaded
    pub fn mark_unloaded(&self, name: &str) {
        let mut models = self.models.lock().unwrap();
        let mut total_used = self.total_memory_used.lock().unwrap();
        
        if let Some(entry) = models.get_mut(name) {
            if entry.state == ModelLoadState::Loaded {
                *total_used -= entry.metadata.size_bytes;
            }
            entry.state = ModelLoadState::Unloaded;
        }
    }
    
    /// Get total memory used
    pub fn get_total_memory_used(&self) -> u64 {
        *self.total_memory_used.lock().unwrap()
    }
    
    /// Get memory budget
    pub fn get_memory_budget(&self) -> u64 {
        self.memory_budget
    }
    
    /// Get all resident models
    pub fn get_resident_models(&self) -> Vec<String> {
        let models = self.models.lock().unwrap();
        models.iter()
            .filter(|(_, e)| e.metadata.role == ModelRole::Resident)
            .map(|(name, _)| name.clone())
            .collect()
    }
    
    /// Get all loaded models
    pub fn get_loaded_models(&self) -> Vec<String> {
        let models = self.models.lock().unwrap();
        models.iter()
            .filter(|(_, e)| e.state == ModelLoadState::Loaded)
            .map(|(name, _)| name.clone())
            .collect()
    }
}

/// Global registry instance
use std::sync::OnceLock;

static MODEL_REGISTRY: OnceLock<Arc<ModelRegistry>> = OnceLock::new();

/// Get global registry instance
pub fn get_registry() -> Arc<ModelRegistry> {
    MODEL_REGISTRY.get_or_init(|| {
        // Default budget: 4 GB (conservative for 16 GB system)
        Arc::new(ModelRegistry::new(4 * 1024 * 1024 * 1024))
    }).clone()
}

/// Initialize registry with budget
pub fn initialize(memory_budget: u64) -> Arc<ModelRegistry> {
    let registry = Arc::new(ModelRegistry::new(memory_budget));
    MODEL_REGISTRY.set(registry.clone()).ok();
    registry
}
