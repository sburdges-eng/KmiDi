"""
Intent IR Emitter - Convert emotion/music state → IntentFrame

This module handles conversion from Python emotion/music state representations
to IntentFrame, with provenance tracking and emission to Rust validator.
"""

from typing import Optional, Dict, Any
from . import IntentFrame, IntentMeta, EmotionState, MusicalIntent, TimeScope, IntentConstraints, IntentProvenance, IntentSource, INTENT_IR_VERSION
import hashlib
import time


class IntentIREmitter:
    """Emitter for Intent IR frames with provenance tracking"""
    
    def __init__(self, session_id: Optional[int] = None):
        """Initialize emitter with optional session ID"""
        self.session_id = session_id or int(time.time() * 1000)  # Use timestamp as default
        self._intent_counter = 0
    
    def _generate_intent_id(self, data: Dict[str, Any]) -> int:
        """Generate stable hash for intent ID"""
        # Create hash from key fields
        hash_input = f"{data.get('emotion', {})}{data.get('music', {})}{data.get('time', {})}"
        hash_bytes = hashlib.sha256(hash_input.encode()).digest()
        return int.from_bytes(hash_bytes[:8], byteorder='big')
    
    def emit_from_emotion(
        self,
        valence: float,
        arousal: float,
        dominance: float,
        discrete_id: Optional[int] = None,
        intensity: float = 1.0,
        confidence: float = 1.0,
        source: IntentSource = IntentSource.ML_AUDIO,
        user_override_weight: float = 0.0,
    ) -> IntentFrame:
        """Emit IntentFrame from emotion VAD coordinates"""
        emotion = EmotionState(
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            discrete_id=discrete_id if discrete_id is not None else -1,
            intensity=intensity,
            confidence=confidence,
        )
        
        # Default musical intent (neutral)
        music = MusicalIntent()
        
        # Generate intent ID
        intent_id = self._generate_intent_id({
            'emotion': {'valence': valence, 'arousal': arousal, 'dominance': dominance},
            'music': {},
            'time': {},
        })
        
        frame = IntentFrame(
            meta=IntentMeta(
                ir_version=INTENT_IR_VERSION,
                intent_id=intent_id,
                session_id=self.session_id,
            ),
            emotion=emotion,
            music=music,
            time=TimeScope(),  # Immediate, open-ended
            constraints=IntentConstraints(),  # No constraints
            provenance=IntentProvenance(
                source=source,
                user_override_weight=user_override_weight,
            ),
        )
        
        return frame
    
    def emit_from_music_params(
        self,
        tempo_bias: float = 0.0,
        rhythmic_density: float = 0.5,
        groove_strength: float = 0.5,
        harmonic_tension: float = 0.5,
        harmonic_motion: float = 0.5,
        mode_preference: int = 0,
        melodic_activity: float = 0.5,
        contour_variance: float = 0.5,
        dynamic_range: float = 0.5,
        texture_density: float = 0.5,
        source: IntentSource = IntentSource.UI_DIRECT,
        user_override_weight: float = 1.0,
    ) -> IntentFrame:
        """Emit IntentFrame from musical parameters"""
        emotion = EmotionState()  # Neutral emotion
        music = MusicalIntent(
            tempo_bias=tempo_bias,
            rhythmic_density=rhythmic_density,
            groove_strength=groove_strength,
            harmonic_tension=harmonic_tension,
            harmonic_motion=harmonic_motion,
            mode_preference=mode_preference,
            melodic_activity=melodic_activity,
            contour_variance=contour_variance,
            dynamic_range=dynamic_range,
            texture_density=texture_density,
        )
        
        # Generate intent ID
        intent_id = self._generate_intent_id({
            'emotion': {},
            'music': {
                'tempo_bias': tempo_bias,
                'rhythmic_density': rhythmic_density,
                'groove_strength': groove_strength,
            },
            'time': {},
        })
        
        frame = IntentFrame(
            meta=IntentMeta(
                ir_version=INTENT_IR_VERSION,
                intent_id=intent_id,
                session_id=self.session_id,
            ),
            emotion=emotion,
            music=music,
            time=TimeScope(),  # Immediate, open-ended
            constraints=IntentConstraints(),  # No constraints
            provenance=IntentProvenance(
                source=source,
                user_override_weight=user_override_weight,
            ),
        )
        
        return frame
    
    def emit_from_full_state(
        self,
        emotion: EmotionState,
        music: MusicalIntent,
        time: Optional[TimeScope] = None,
        constraints: Optional[IntentConstraints] = None,
        source: IntentSource = IntentSource.ML_TEXT,
        user_override_weight: float = 0.0,
    ) -> IntentFrame:
        """Emit IntentFrame from full state"""
        # Generate intent ID
        intent_id = self._generate_intent_id({
            'emotion': {
                'valence': emotion.valence,
                'arousal': emotion.arousal,
                'dominance': emotion.dominance,
            },
            'music': {
                'tempo_bias': music.tempo_bias,
                'rhythmic_density': music.rhythmic_density,
            },
            'time': {
                'start_bar': time.start_bar if time else -1,
                'end_bar': time.end_bar if time else -1,
            },
        })
        
        frame = IntentFrame(
            meta=IntentMeta(
                ir_version=INTENT_IR_VERSION,
                intent_id=intent_id,
                session_id=self.session_id,
            ),
            emotion=emotion,
            music=music,
            time=time or TimeScope(),
            constraints=constraints or IntentConstraints(),
            provenance=IntentProvenance(
                source=source,
                user_override_weight=user_override_weight,
            ),
        )
        
        return frame
    
    def emit_from_dict(self, data: Dict[str, Any], source: IntentSource = IntentSource.ML_TEXT) -> IntentFrame:
        """Emit IntentFrame from dictionary (for JSON deserialization)"""
        # This is a convenience method for creating frames from external data
        # The data should match the IntentFrame structure
        emotion_data = data.get('emotion', {})
        music_data = data.get('music', {})
        time_data = data.get('time', {})
        constraints_data = data.get('constraints', {})
        provenance_data = data.get('provenance', {})
        
        emotion = EmotionState(**emotion_data)
        music = MusicalIntent(**music_data)
        time = TimeScope(**time_data)
        constraints = IntentConstraints(**constraints_data)
        provenance = IntentProvenance(
            source=IntentSource(provenance_data.get('source', source)),
            user_override_weight=provenance_data.get('user_override_weight', 0.0),
        )
        
        return self.emit_from_full_state(
            emotion=emotion,
            music=music,
            time=time,
            constraints=constraints,
            source=provenance.source,
            user_override_weight=provenance.user_override_weight,
        )
