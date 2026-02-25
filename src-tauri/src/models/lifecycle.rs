//! Model Lifecycle Manager - Background loading, hot-swap, graceful degradation

use super::registry::{ModelRegistry, ModelRole, get_registry};
use std::sync::Arc;
use tokio::task;

/// Model lifecycle manager
pub struct ModelLifecycleManager {
    registry: Arc<ModelRegistry>,
}

impl ModelLifecycleManager {
    pub fn new(registry: Arc<ModelRegistry>) -> Self {
        Self { registry }
    }
    
    /// Load model on background thread
    pub async fn load_model_async(&self, name: String) -> Result<(), String> {
        // Check if can load
        if !self.registry.can_load(&name) {
            return Err(format!("Cannot load {}: memory budget exceeded", name));
        }
        
        // Check role
        let metadata = self.registry.get_metadata(&name)
            .ok_or_else(|| format!("Model {} not registered", name))?;
        
        if metadata.role == ModelRole::Offline {
            return Err(format!("Model {} is offline-only", name));
        }
        
        // Mark as loading
        self.registry.mark_loading(&name);
        
        // Simulate loading on background thread
        // In real implementation, this would actually load the model
        task::spawn_blocking(move || {
            // Simulate load time
            std::thread::sleep(std::time::Duration::from_millis(100));
        }).await.unwrap();
        
        // Mark as loaded
        self.registry.mark_loaded(&name);
        
        Ok(())
    }
    
    /// Unload model
    pub fn unload_model(&self, name: &str) -> Result<(), String> {
        let metadata = self.registry.get_metadata(name)
            .ok_or_else(|| format!("Model {} not registered", name))?;
        
        // Never evict resident models
        if metadata.role == ModelRole::Resident {
            return Err(format!("Cannot unload resident model {}", name));
        }
        
        self.registry.mark_unloaded(name);
        Ok(())
    }
    
    /// Hot-swap model (unload old, load new)
    pub async fn hot_swap(&self, old_name: &str, new_name: &str) -> Result<(), String> {
        // Unload old (if not resident)
        if let Err(e) = self.unload_model(old_name) {
            // Ignore errors for resident models
            if !e.contains("resident") {
                return Err(e);
            }
        }
        
        // Load new
        self.load_model_async(new_name.to_string()).await
    }
    
    /// Ensure resident models are loaded
    pub async fn ensure_resident_models(&self) -> Result<(), String> {
        let resident_models = self.registry.get_resident_models();
        
        for name in resident_models {
            if self.registry.can_load(&name) {
                if let Err(e) = self.load_model_async(name.clone()).await {
                    return Err(format!("Failed to load resident model {}: {}", name, e));
                }
            }
        }
        
        Ok(())
    }
}

/// Global lifecycle manager
use std::sync::OnceLock;

static LIFECYCLE_MANAGER: OnceLock<Arc<ModelLifecycleManager>> = OnceLock::new();

/// Get global lifecycle manager
pub fn get_lifecycle_manager() -> Arc<ModelLifecycleManager> {
    LIFECYCLE_MANAGER.get_or_init(|| {
        Arc::new(ModelLifecycleManager::new(get_registry()))
    }).clone()
}
