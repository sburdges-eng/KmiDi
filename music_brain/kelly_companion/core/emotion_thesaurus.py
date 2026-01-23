"""
Kelly Emotion Thesaurus - Complete 216-node emotion network with musical mappings.

Loads emotion data from JSON files and provides:
- Word → Emotion lookup (for interrogator)
- Emotion → Musical Parameters mapping
- Emotion space navigation (nearby emotions, transitions)
- Intensity-aware musical attribute scaling
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# ENUMS & CONSTANTS
# =============================================================================

class EmotionCategory(Enum):
    """Primary emotion categories (Plutchik's wheel)."""
    JOY = "joy"
    SADNESS = "sad"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"


class Mode(Enum):
    """Musical modes ordered by brightness (dark → bright)."""
    LOCRIAN = "locrian"
    PHRYGIAN = "phrygian"
    AEOLIAN = "aeolian"
    DORIAN = "dorian"
    MIXOLYDIAN = "mixolydian"
    IONIAN = "ionian"
    LYDIAN = "lydian"


# Valence mapping: category → base valence (-1 to 1)
CATEGORY_VALENCE = {
    EmotionCategory.JOY: 0.8,
    EmotionCategory.SADNESS: -0.7,
    EmotionCategory.ANGER: -0.6,
    EmotionCategory.FEAR: -0.5,
    EmotionCategory.SURPRISE: 0.1,  # Neutral, depends on context
    EmotionCategory.DISGUST: -0.4,
    EmotionCategory.TRUST: 0.5,
    EmotionCategory.ANTICIPATION: 0.3,
}

# Arousal mapping: category → base arousal (0 to 1)
CATEGORY_AROUSAL = {
    EmotionCategory.JOY: 0.7,
    EmotionCategory.SADNESS: 0.3,
    EmotionCategory.ANGER: 0.9,
    EmotionCategory.FEAR: 0.8,
    EmotionCategory.SURPRISE: 0.8,
    EmotionCategory.DISGUST: 0.5,
    EmotionCategory.TRUST: 0.4,
    EmotionCategory.ANTICIPATION: 0.6,
}

# Intensity tier → numeric value
INTENSITY_VALUES = {
    "1_subtle": 0.2,
    "2_mild": 0.4,
    "3_moderate": 0.6,
    "4_intense": 0.8,
    "5_overwhelming": 1.0,
}


# =============================================================================
# MUSICAL PARAMETER MAPPINGS
# =============================================================================

@dataclass
class MusicalAttributes:
    """Musical parameters derived from emotional state."""
    # Tempo
    tempo_base: int = 100
    tempo_range: Tuple[int, int] = (90, 110)
    tempo_volatility: float = 0.1  # How much tempo can drift
    
    # Harmony
    mode: Mode = Mode.IONIAN
    modes_allowed: List[Mode] = field(default_factory=lambda: [Mode.IONIAN])
    borrowed_chord_probability: float = 0.0
    dissonance_level: float = 0.0  # 0-1
    
    # Rhythm
    swing_amount: float = 0.0  # 0-1
    syncopation: float = 0.0  # 0-1
    note_density: float = 0.5  # 0-1
    rest_frequency: float = 0.2  # 0-1
    
    # Dynamics
    velocity_base: int = 80
    velocity_range: Tuple[int, int] = (60, 100)
    dynamic_contour: str = "steady"  # steady, swell, decay, pulse
    accent_strength: float = 0.3
    
    # Articulation
    legato: float = 0.5  # 0=staccato, 1=full legato
    attack_sharpness: float = 0.5  # 0=soft, 1=sharp
    release_sustain: float = 0.5  # 0=short, 1=sustained
    
    # Expression
    vibrato_depth: float = 0.0
    pitch_drift: float = 0.0  # Slight detuning for rawness
    humanize_timing: float = 0.1  # Timing imperfection
    humanize_velocity: float = 0.1  # Velocity imperfection
    
    # Space/Production
    reverb_send: float = 0.3
    stereo_width: float = 0.5
    register: str = "mid"  # low, mid, high
    
    # Rule-breaking directives
    rule_breaks: List[str] = field(default_factory=list)


# Category → Musical DNA
CATEGORY_MUSICAL_DNA: Dict[EmotionCategory, Dict[str, Any]] = {
    EmotionCategory.JOY: {
        "tempo_range": (110, 140),
        "modes": [Mode.IONIAN, Mode.LYDIAN, Mode.MIXOLYDIAN],
        "swing": 0.2,
        "legato": 0.6,
        "velocity_range": (70, 110),
        "dynamic_contour": "swell",
        "register": "mid-high",
        "reverb": 0.3,
        "rule_breaks": ["add_passing_tones", "anticipate_beats"],
    },
    EmotionCategory.SADNESS: {
        "tempo_range": (50, 80),
        "modes": [Mode.AEOLIAN, Mode.DORIAN, Mode.PHRYGIAN],
        "swing": 0.0,
        "legato": 0.8,
        "velocity_range": (40, 80),
        "dynamic_contour": "decay",
        "register": "low-mid",
        "reverb": 0.6,
        "rule_breaks": ["suspend_resolution", "modal_mixture", "rubato"],
    },
    EmotionCategory.ANGER: {
        "tempo_range": (120, 180),
        "modes": [Mode.PHRYGIAN, Mode.LOCRIAN, Mode.AEOLIAN],
        "swing": 0.0,
        "legato": 0.2,
        "velocity_range": (90, 127),
        "dynamic_contour": "pulse",
        "register": "low",
        "reverb": 0.1,
        "rule_breaks": ["add_dissonance", "metric_displacement", "cluster_voicing"],
    },
    EmotionCategory.FEAR: {
        "tempo_range": (80, 140),
        "modes": [Mode.LOCRIAN, Mode.PHRYGIAN],
        "swing": 0.0,
        "legato": 0.4,
        "velocity_range": (50, 100),
        "dynamic_contour": "erratic",
        "register": "extreme",  # Very low or very high
        "reverb": 0.5,
        "rule_breaks": ["tritone_substitution", "avoid_root", "unstable_voicing"],
    },
    EmotionCategory.SURPRISE: {
        "tempo_range": (100, 130),
        "modes": [Mode.LYDIAN, Mode.MIXOLYDIAN],
        "swing": 0.1,
        "legato": 0.5,
        "velocity_range": (60, 120),
        "dynamic_contour": "spike",
        "register": "mid-high",
        "reverb": 0.4,
        "rule_breaks": ["unexpected_chord", "meter_change", "register_jump"],
    },
    EmotionCategory.DISGUST: {
        "tempo_range": (70, 100),
        "modes": [Mode.LOCRIAN, Mode.PHRYGIAN],
        "swing": 0.0,
        "legato": 0.3,
        "velocity_range": (60, 90),
        "dynamic_contour": "steady",
        "register": "low-mid",
        "reverb": 0.2,
        "rule_breaks": ["chromatic_slide", "cluster_voicing", "avoid_resolution"],
    },
    EmotionCategory.TRUST: {
        "tempo_range": (70, 100),
        "modes": [Mode.IONIAN, Mode.LYDIAN],
        "swing": 0.1,
        "legato": 0.7,
        "velocity_range": (60, 90),
        "dynamic_contour": "steady",
        "register": "mid",
        "reverb": 0.4,
        "rule_breaks": ["pedal_tone", "plagal_cadence"],
    },
    EmotionCategory.ANTICIPATION: {
        "tempo_range": (100, 130),
        "modes": [Mode.DORIAN, Mode.MIXOLYDIAN],
        "swing": 0.2,
        "legato": 0.5,
        "velocity_range": (70, 100),
        "dynamic_contour": "building",
        "register": "mid",
        "reverb": 0.3,
        "rule_breaks": ["delay_resolution", "add_suspension", "dominant_pedal"],
    },
}


# =============================================================================
# EMOTION NODE
# =============================================================================

@dataclass
class EmotionNode:
    """A single emotion in the thesaurus with full musical mapping."""
    # Identity
    id: int
    name: str
    category: EmotionCategory
    
    # Hierarchy
    sub_emotion: str  # e.g., "grief", "euphoria"
    sub_sub_emotion: str  # e.g., "bereaved", "elated"
    intensity_tier: str  # e.g., "3_moderate"
    
    # Dimensional position
    valence: float  # -1 to 1
    arousal: float  # 0 to 1
    intensity: float  # 0 to 1
    
    # Synonyms (the actual words users might say)
    words: List[str] = field(default_factory=list)
    
    # Description from JSON
    description: str = ""
    
    # Musical mapping
    musical: MusicalAttributes = field(default_factory=MusicalAttributes)
    
    # Graph connections
    related_ids: List[int] = field(default_factory=list)
    opposite_ids: List[int] = field(default_factory=list)
    
    def distance_to(self, other: 'EmotionNode') -> float:
        """Calculate emotional distance to another node."""
        return (
            (self.valence - other.valence) ** 2 +
            (self.arousal - other.arousal) ** 2 +
            (self.intensity - other.intensity) ** 2
        ) ** 0.5


# =============================================================================
# MAIN THESAURUS CLASS
# =============================================================================

class EmotionThesaurus:
    """
    The 216-node emotion network.
    
    Provides:
    - Word lookup: "devastated" → EmotionNode
    - Emotion navigation: find nearby/opposite emotions
    - Musical parameter generation
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize thesaurus from JSON files.
        
        Args:
            data_dir: Directory containing emotion JSON files.
                      Defaults to same directory as this file.
        """
        self.nodes: Dict[int, EmotionNode] = {}
        self.word_index: Dict[str, int] = {}  # word → node_id
        self.category_index: Dict[EmotionCategory, List[int]] = {}
        self._next_id = 0
        
        if data_dir is None:
            data_dir = Path(__file__).parent
        
        self._load_from_json(data_dir)
        self._build_relationships()
    
    def _load_from_json(self, data_dir: Path) -> None:
        """Load all emotion JSON files."""
        json_files = {
            "joy.json": EmotionCategory.JOY,
            "sad.json": EmotionCategory.SADNESS,
            "anger.json": EmotionCategory.ANGER,
            "fear.json": EmotionCategory.FEAR,
            "surprise.json": EmotionCategory.SURPRISE,
            "disgust.json": EmotionCategory.DISGUST,
        }
        
        for filename, category in json_files.items():
            filepath = data_dir / filename
            if filepath.exists():
                self._parse_emotion_file(filepath, category)
            else:
                print(f"Warning: {filename} not found at {filepath}")
        
        print(f"Loaded {len(self.nodes)} emotion nodes, {len(self.word_index)} words indexed")
    
    def _parse_emotion_file(self, filepath: Path, category: EmotionCategory) -> None:
        """Parse a single emotion JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        base_valence = CATEGORY_VALENCE[category]
        base_arousal = CATEGORY_AROUSAL[category]
        
        for sub_name, sub_data in data.get("sub_emotions", {}).items():
            sub_description = sub_data.get("description", "")
            
            for subsub_name, subsub_data in sub_data.get("sub_sub_emotions", {}).items():
                subsub_description = subsub_data.get("description", "")
                
                for tier_name, words in subsub_data.get("intensity_tiers", {}).items():
                    intensity = INTENSITY_VALUES.get(tier_name, 0.5)
                    
                    # Calculate dimensional values
                    # Intensity affects arousal and valence extremity
                    valence = base_valence * (0.5 + intensity * 0.5)
                    arousal = base_arousal * (0.7 + intensity * 0.3)
                    arousal = min(1.0, max(0.0, arousal))
                    
                    # Create node
                    node = EmotionNode(
                        id=self._next_id,
                        name=f"{subsub_name}_{tier_name}",
                        category=category,
                        sub_emotion=sub_name,
                        sub_sub_emotion=subsub_name,
                        intensity_tier=tier_name,
                        valence=valence,
                        arousal=arousal,
                        intensity=intensity,
                        words=words if isinstance(words, list) else [words],
                        description=subsub_description or sub_description,
                        musical=self._generate_musical_attributes(
                            category, intensity, valence, arousal
                        ),
                    )
                    
                    # Store node
                    self.nodes[self._next_id] = node
                    
                    # Index words
                    for word in node.words:
                        word_lower = word.lower().strip()
                        self.word_index[word_lower] = self._next_id
                    
                    # Also index the subsub name
                    self.word_index[subsub_name.lower()] = self._next_id
                    
                    # Category index
                    if category not in self.category_index:
                        self.category_index[category] = []
                    self.category_index[category].append(self._next_id)
                    
                    self._next_id += 1
    
    def _generate_musical_attributes(
        self,
        category: EmotionCategory,
        intensity: float,
        valence: float,
        arousal: float,
    ) -> MusicalAttributes:
        """Generate musical parameters from emotional dimensions."""
        dna = CATEGORY_MUSICAL_DNA.get(category, CATEGORY_MUSICAL_DNA[EmotionCategory.JOY])
        
        # Calculate tempo from arousal and intensity
        tempo_low, tempo_high = dna["tempo_range"]
        tempo_base = int(tempo_low + (tempo_high - tempo_low) * arousal)
        tempo_range = (
            max(40, tempo_base - int(20 * intensity)),
            min(200, tempo_base + int(20 * intensity)),
        )
        
        # Select mode based on valence
        modes = dna["modes"]
        if valence > 0.3:
            mode = modes[0] if modes else Mode.IONIAN
        elif valence < -0.3:
            mode = modes[-1] if modes else Mode.AEOLIAN
        else:
            mode = modes[len(modes) // 2] if modes else Mode.DORIAN
        
        # Velocity from intensity
        vel_low, vel_high = dna["velocity_range"]
        velocity_base = int(vel_low + (vel_high - vel_low) * intensity)
        
        # Dissonance from negative valence + high arousal
        dissonance = max(0, -valence) * arousal * intensity
        
        # Borrowed chord probability from intensity
        borrowed_prob = intensity * 0.3 if abs(valence) > 0.5 else 0.1
        
        return MusicalAttributes(
            tempo_base=tempo_base,
            tempo_range=tempo_range,
            tempo_volatility=intensity * 0.2,
            mode=mode,
            modes_allowed=modes,
            borrowed_chord_probability=borrowed_prob,
            dissonance_level=dissonance,
            swing_amount=dna["swing"] * (1 - intensity * 0.5),
            syncopation=arousal * 0.5,
            note_density=0.3 + arousal * 0.5,
            rest_frequency=0.3 - arousal * 0.2,
            velocity_base=velocity_base,
            velocity_range=(
                max(1, velocity_base - int(30 * intensity)),
                min(127, velocity_base + int(20 * intensity)),
            ),
            dynamic_contour=dna["dynamic_contour"],
            accent_strength=intensity * 0.5,
            legato=dna["legato"] * (1 + (1 - arousal) * 0.3),
            attack_sharpness=arousal * intensity,
            release_sustain=1 - arousal * 0.5,
            vibrato_depth=intensity * 0.3 if valence < 0 else intensity * 0.1,
            pitch_drift=intensity * 0.02 if valence < -0.5 else 0,
            humanize_timing=0.05 + intensity * 0.1,
            humanize_velocity=0.05 + intensity * 0.15,
            reverb_send=dna["reverb"],
            stereo_width=0.5 + intensity * 0.3,
            register=dna["register"],
            rule_breaks=dna["rule_breaks"] if intensity > 0.5 else [],
        )
    
    def _build_relationships(self) -> None:
        """Build relationship graph between nodes."""
        nodes_list = list(self.nodes.values())
        
        for node in nodes_list:
            # Find related (nearby in emotional space)
            nearby = self.get_nearby_emotions(node.id, threshold=0.4, limit=5)
            node.related_ids = [n.id for n in nearby]
            
            # Find opposites (far in valence, similar arousal)
            opposites = []
            for other in nodes_list:
                if other.id == node.id:
                    continue
                valence_diff = abs(node.valence - (-other.valence))
                arousal_diff = abs(node.arousal - other.arousal)
                if valence_diff < 0.3 and arousal_diff < 0.3:
                    opposites.append(other.id)
            node.opposite_ids = opposites[:3]
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    def lookup(self, word: str) -> Optional[EmotionNode]:
        """
        Look up emotion by word.
        
        Args:
            word: Any emotion word (e.g., "devastated", "joyful", "anxious")
            
        Returns:
            EmotionNode if found, None otherwise
        """
        word_lower = word.lower().strip()
        node_id = self.word_index.get(word_lower)
        if node_id is not None:
            return self.nodes.get(node_id)
        
        # Try partial match
        for indexed_word, nid in self.word_index.items():
            if word_lower in indexed_word or indexed_word in word_lower:
                return self.nodes.get(nid)
        
        return None
    
    def get(self, node_id: int) -> Optional[EmotionNode]:
        """Get emotion by ID."""
        return self.nodes.get(node_id)
    
    def get_by_name(self, name: str) -> Optional[EmotionNode]:
        """Get emotion by node name."""
        for node in self.nodes.values():
            if node.name.lower() == name.lower():
                return node
        return None
    
    def get_by_category(self, category: EmotionCategory) -> List[EmotionNode]:
        """Get all emotions in a category."""
        ids = self.category_index.get(category, [])
        return [self.nodes[i] for i in ids]
    
    def get_nearby_emotions(
        self,
        node_id: int,
        threshold: float = 0.3,
        limit: int = 10,
    ) -> List[EmotionNode]:
        """Find emotions near the given emotion in emotional space."""
        source = self.nodes.get(node_id)
        if not source:
            return []
        
        distances = []
        for node in self.nodes.values():
            if node.id == node_id:
                continue
            dist = source.distance_to(node)
            if dist < threshold:
                distances.append((dist, node))
        
        distances.sort(key=lambda x: x[0])
        return [node for _, node in distances[:limit]]
    
    def find_transition_path(
        self,
        from_id: int,
        to_id: int,
        steps: int = 3,
    ) -> List[EmotionNode]:
        """
        Find a smooth emotional transition path.
        
        Useful for Side A → Side B transitions.
        """
        start = self.nodes.get(from_id)
        end = self.nodes.get(to_id)
        if not start or not end:
            return []
        
        path = [start]
        
        # Linear interpolation in emotional space
        for i in range(1, steps):
            t = i / steps
            target_valence = start.valence + (end.valence - start.valence) * t
            target_arousal = start.arousal + (end.arousal - start.arousal) * t
            target_intensity = start.intensity + (end.intensity - start.intensity) * t
            
            # Find closest node to interpolated point
            best_node = None
            best_dist = float('inf')
            
            for node in self.nodes.values():
                if node.id in [n.id for n in path]:
                    continue
                dist = (
                    (node.valence - target_valence) ** 2 +
                    (node.arousal - target_arousal) ** 2 +
                    (node.intensity - target_intensity) ** 2
                ) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_node = node
            
            if best_node:
                path.append(best_node)
        
        path.append(end)
        return path
    
    def get_musical_params(self, word: str) -> Optional[MusicalAttributes]:
        """Convenience: word → musical parameters directly."""
        node = self.lookup(word)
        return node.musical if node else None
    
    def blend_emotions(
        self,
        emotions: List[Tuple[str, float]],  # [(word, weight), ...]
    ) -> MusicalAttributes:
        """
        Blend multiple emotions into combined musical parameters.
        
        Args:
            emotions: List of (word, weight) tuples
            
        Returns:
            Blended MusicalAttributes
        """
        nodes_weights = []
        total_weight = 0
        
        for word, weight in emotions:
            node = self.lookup(word)
            if node:
                nodes_weights.append((node, weight))
                total_weight += weight
        
        if not nodes_weights:
            return MusicalAttributes()
        
        # Normalize weights
        nodes_weights = [(n, w / total_weight) for n, w in nodes_weights]
        
        # Blend dimensional values
        valence = sum(n.valence * w for n, w in nodes_weights)
        arousal = sum(n.arousal * w for n, w in nodes_weights)
        intensity = sum(n.intensity * w for n, w in nodes_weights)
        
        # Use highest-weight node's category for DNA
        primary_node = max(nodes_weights, key=lambda x: x[1])[0]
        
        return self._generate_musical_attributes(
            primary_node.category, intensity, valence, arousal
        )
    
    def search(self, query: str, limit: int = 10) -> List[EmotionNode]:
        """
        Fuzzy search for emotions matching query.
        
        Searches words, names, descriptions.
        """
        query_lower = query.lower()
        matches = []
        
        for node in self.nodes.values():
            score = 0
            
            # Check words
            for word in node.words:
                if query_lower in word.lower():
                    score += 10
                elif word.lower() in query_lower:
                    score += 5
            
            # Check name
            if query_lower in node.name.lower():
                score += 8
            
            # Check description
            if query_lower in node.description.lower():
                score += 3
            
            if score > 0:
                matches.append((score, node))
        
        matches.sort(key=lambda x: -x[0])
        return [node for _, node in matches[:limit]]
    
    def stats(self) -> Dict[str, Any]:
        """Return statistics about the thesaurus."""
        return {
            "total_nodes": len(self.nodes),
            "total_words": len(self.word_index),
            "categories": {
                cat.value: len(ids) 
                for cat, ids in self.category_index.items()
            },
            "valence_range": (
                min(n.valence for n in self.nodes.values()),
                max(n.valence for n in self.nodes.values()),
            ),
            "arousal_range": (
                min(n.arousal for n in self.nodes.values()),
                max(n.arousal for n in self.nodes.values()),
            ),
        }
    
    def __len__(self) -> int:
        return len(self.nodes)
    
    def __contains__(self, word: str) -> bool:
        return word.lower() in self.word_index


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_default_thesaurus: Optional[EmotionThesaurus] = None


def get_thesaurus(data_dir: Optional[Path] = None) -> EmotionThesaurus:
    """Get or create the default thesaurus instance."""
    global _default_thesaurus
    if _default_thesaurus is None:
        _default_thesaurus = EmotionThesaurus(data_dir)
    return _default_thesaurus


def emotion_to_music(word: str) -> Optional[MusicalAttributes]:
    """Quick lookup: word → musical parameters."""
    return get_thesaurus().get_musical_params(word)


def find_emotion(word: str) -> Optional[EmotionNode]:
    """Quick lookup: word → emotion node."""
    return get_thesaurus().lookup(word)


# =============================================================================
# CLI / TEST
# =============================================================================

if __name__ == "__main__":
    import sys
    
    # Initialize from current directory or command line arg
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    
    thesaurus = EmotionThesaurus(data_dir)
    
    print("\n=== EMOTION THESAURUS STATS ===")
    stats = thesaurus.stats()
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Total indexed words: {stats['total_words']}")
    print(f"Categories: {stats['categories']}")
    
    # Test lookups
    test_words = ["devastated", "joyful", "anxious", "furious", "content", "terrified"]
    
    print("\n=== WORD LOOKUPS ===")
    for word in test_words:
        node = thesaurus.lookup(word)
        if node:
            print(f"\n'{word}' → {node.name}")
            print(f"  Category: {node.category.value}")
            print(f"  Valence: {node.valence:.2f}, Arousal: {node.arousal:.2f}, Intensity: {node.intensity:.2f}")
            print(f"  Tempo: {node.musical.tempo_base} BPM ({node.musical.tempo_range})")
            print(f"  Mode: {node.musical.mode.value}")
            print(f"  Rule breaks: {node.musical.rule_breaks}")
        else:
            print(f"\n'{word}' → NOT FOUND")
    
    # Test transition
    print("\n=== EMOTION TRANSITION ===")
    grief = thesaurus.lookup("devastated")
    peace = thesaurus.lookup("content")
    if grief and peace:
        path = thesaurus.find_transition_path(grief.id, peace.id, steps=4)
        print(f"Path from '{grief.name}' to '{peace.name}':")
        for i, node in enumerate(path):
            print(f"  {i+1}. {node.name} (v={node.valence:.2f}, a={node.arousal:.2f})")
    
    # Test blend
    print("\n=== EMOTION BLEND ===")
    blend = thesaurus.blend_emotions([("sad", 0.6), ("angry", 0.4)])
    print(f"60% sad + 40% angry:")
    print(f"  Tempo: {blend.tempo_base} BPM")
    print(f"  Mode: {blend.mode.value}")
    print(f"  Dissonance: {blend.dissonance_level:.2f}")
