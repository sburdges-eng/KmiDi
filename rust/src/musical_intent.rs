use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IntentAspect {
    Harmonic,
    Rhythmic,
    Dynamic,
    Tempo,
}

#[derive(Debug, Clone)]
pub struct IntentComponent {
    pub aspect: IntentAspect,
    pub magnitude: f64,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct MusicalIntent {
    components: HashMap<IntentAspect, IntentComponent>,
}

impl MusicalIntent {
    pub fn new() -> Self {
        Self {
            components: HashMap::new(),
        }
    }

    pub fn with_components(components: impl IntoIterator<Item = IntentComponent>) -> Self {
        let mut intent = Self::new();
        for c in components {
            intent.components.insert(c.aspect, c);
        }
        intent
    }

    pub fn update(&mut self, other: &MusicalIntent) {
        for (aspect, comp) in &other.components {
            self.components.insert(*aspect, comp.clone());
        }
    }

    pub fn merge(&mut self, src: &MusicalIntent) {
        self.update(src);
    }

    pub fn serialize(&self) -> String {
        let mut map = serde_json::Map::new();
        for (aspect, comp) in &self.components {
            let key = match aspect {
                IntentAspect::Harmonic => "harmonic",
                IntentAspect::Rhythmic => "rhythmic",
                IntentAspect::Dynamic => "dynamic",
                IntentAspect::Tempo => "tempo",
            };
            map.insert(key.to_string(), serde_json::json!(comp.magnitude));
        }
        serde_json::to_string(&map).unwrap_or_else(|_| "{}".to_string())
    }

    pub fn deserialize(json: &str) -> Self {
        let map: HashMap<String, f64> = serde_json::from_str(json).unwrap_or_default();
        let mut intent = Self::new();
        for (k, v) in map {
            let aspect = match k.as_str() {
                "harmonic" => IntentAspect::Harmonic,
                "rhythmic" => IntentAspect::Rhythmic,
                "dynamic" => IntentAspect::Dynamic,
                "tempo" => IntentAspect::Tempo,
                _ => continue,
            };
            intent.components.insert(
                aspect,
                IntentComponent {
                    aspect,
                    magnitude: v,
                    reason: "deserialized".to_string(),
                },
            );
        }
        intent
    }

    pub fn distance(&self, other: &MusicalIntent) -> f64 {
        let mut dist = 0.0;
        for (aspect, comp) in &self.components {
            let other_mag = other
                .components
                .get(aspect)
                .map(|c| c.magnitude)
                .unwrap_or(0.0);
            dist += (comp.magnitude - other_mag).abs();
        }
        dist
    }

    pub fn has_aspect(&self, aspect: IntentAspect) -> bool {
        self.components.contains_key(&aspect)
    }

    pub fn get_magnitude(&self, aspect: IntentAspect) -> f64 {
        self.components
            .get(&aspect)
            .map(|c| c.magnitude)
            .unwrap_or(0.0)
    }
}

impl Default for MusicalIntent {
    fn default() -> Self {
        Self::new()
    }
}
