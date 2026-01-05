# flake8: noqa
# ruff: noqa: E501
"""
Music Brain Emotion API - Clean interface for emotion-to-music generation.

Provides both declarative and fluent API styles for mapping emotional intent
to musical parameters and DAW mixer automation.

Philosophy: "Interrogate Before Generate" - Emotional intent drives technical choices.

Usage (Declarative):
    brain = MusicBrain()
    music = brain.generate_from_intent(intent)
    brain.export_to_logic(music, "output.mid")

Usage (Fluent):
    brain = MusicBrain()
    result = (brain.process("grief and loss")
                   .map_to_mixer()
                   .export_logic("output_automation.json"))
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

# Import existing modules
from music_brain.session.intent_schema import (
    CompleteSongIntent,
    SongRoot,
    SongIntent,
    TechnicalConstraints,
    SystemDirective,
    suggest_rule_break,
)
from music_brain.data.emotional_mapping import (
    EmotionalState,
    MusicalParameters,
    Valence,
    Arousal,
    TimingFeel,
    get_parameters_for_state,
)
from music_brain.daw.mixer_params import (
    MixerParameters,
    EmotionMapper,
    export_to_logic_automation,
    export_mixer_settings,
)
from music_brain.production.emotion_production import (
    ProductionPreset,
    EmotionProductionMapper,
)
from music_brain.production.dynamics_engine import DynamicsEngine
from music_brain.production.drum_humanizer import DrumHumanizer
from music_brain.groove.groove_engine import GrooveSettings, settings_from_intent

# New multimodal emotion components
from music_brain.emotion.text_emotion_parser import TextEmotionParser, ParsedEmotion
from music_brain.emotion.multimodal_emotion import (
    MultimodalEmotionModel,
    prepare_text_features,
    MultimodalEmotionOutput,
)

# Optional advanced components (lazy loaded)
HAS_SPECTOCLOUD = False
HAS_SUGGESTIONS = False
HAS_ARRANGEMENT = False

try:
    from music_brain.visualization.spectocloud import Spectocloud
    HAS_SPECTOCLOUD = True
except ImportError:
    pass

try:
    from music_brain.intelligence.suggestion_engine import SuggestionEngine
    HAS_SUGGESTIONS = True
except ImportError:
    pass

try:
    from music_brain.generative.arrangement import ArrangementGenerator
    HAS_ARRANGEMENT = True
except ImportError:
    pass


# Intent examples for reference
INTENT_EXAMPLES = {
    "therapeutic": "Create calming music for anxiety",
    "workout": "Energetic, driving beat for running",
    "grief_processing": "Melancholic, introspective piece for loss",
    "anger_release": "Cathartic, heavy track for rage",
    "misdirection": "Surface happy, undertow sad (emotional contrast)",
    "nostalgia": "Warm, lo-fi piece for memory processing",
    "hope": "Uplifting, building toward resolution",
    "tension": "Building suspense and unease",
}


@dataclass
class GeneratedMusic:
    """Complete music generation result."""
    emotional_state: EmotionalState
    musical_params: MusicalParameters
    mixer_params: MixerParameters
    production_preset: Optional[ProductionPreset] = None
    groove_settings: Optional[GrooveSettings] = None
    intent: Optional[CompleteSongIntent] = None
    midi_path: Optional[str] = None
    automation_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "emotional_state": {
                "valence": self.emotional_state.valence.value if isinstance(self.emotional_state.valence, Valence) else self.emotional_state.valence,
                "arousal": self.emotional_state.arousal.value if isinstance(self.emotional_state.arousal, Arousal) else self.emotional_state.arousal,
                "primary_emotion": self.emotional_state.primary_emotion,
                "secondary_emotions": self.emotional_state.secondary_emotions,
            },
            "musical_params": {
                "tempo_suggested": self.musical_params.tempo_suggested,
                "tempo_range": (self.musical_params.tempo_min, self.musical_params.tempo_max),
                "dissonance": self.musical_params.dissonance,
                "timing_feel": self.musical_params.timing_feel.value,
                "density_suggested": self.musical_params.density_suggested,
                "space_probability": self.musical_params.space_probability,
            },
            "mixer_params": self.mixer_params.to_dict(),
            "production_preset": (
                self.production_preset.as_dict() if self.production_preset else None
            ),
            "groove_settings": (
                self.groove_settings.to_dict() if self.groove_settings else None
            ),
            "paths": {
                "midi": self.midi_path,
                "automation": self.automation_path,
            },
        }

    def summary(self) -> str:
        """Get a human-readable summary."""
        lines = [
            "=" * 60,
            "GENERATED MUSIC SUMMARY",
            "=" * 60,
            "",
            f"Primary Emotion: {self.emotional_state.primary_emotion}",
            f"Valence: {self.emotional_state.valence}",
            f"Arousal: {self.emotional_state.arousal}",
            "",
            "Musical Parameters:",
            f"  Tempo: {self.musical_params.tempo_suggested} BPM",
            f"  Timing Feel: {self.musical_params.timing_feel.value}",
            f"  Dissonance: {self.musical_params.dissonance:.0%}",
            f"  Density: {self.musical_params.density_suggested}",
            "",
            "Mixer Settings:",
            f"  Description: {self.mixer_params.description}",
            f"  Reverb: {self.mixer_params.reverb_mix:.0%} mix, "
            f"{self.mixer_params.reverb_decay:.1f}s decay",
            f"  Compression: {self.mixer_params.compression_ratio:.1f}:1",
            "  Saturation: "
            f"{self.mixer_params.saturation:.0%} "
            f"({self.mixer_params.saturation_type})",
            "",
            "Output Files:",
            f"  MIDI: {self.midi_path or 'Not generated'}",
            f"  Automation: {self.automation_path or 'Not generated'}",
            "",
            "=" * 60,
        ]
        return "\n".join(lines)


class MusicBrain:
    """
    Main Music Brain API for emotion-to-music generation.

    Provides a clean, consistent interface for all emotion-to-music operations,
    making it easier to integrate with desktop apps, web services, or CLI tools.
    """

    def __init__(self, use_neural: bool = True) -> None:
        self.emotion_mapper = EmotionMapper()
        self.production_mapper = EmotionProductionMapper()
        self.dynamics_engine = DynamicsEngine()
        self.drum_humanizer = self._build_humanizer()
        self._emotion_keywords = self._build_emotion_keywords()

        # Initialize neural emotion components
        self.use_neural = use_neural
        self.text_parser: Optional[TextEmotionParser] = None
        self.multimodal_model: Optional[MultimodalEmotionModel] = None
        self._device = "cpu"

        if use_neural:
            self._init_neural_components()

    def _init_neural_components(self) -> None:
        """Initialize neural emotion recognition components."""
        import torch
        from pathlib import Path

        # Detect device
        if torch.backends.mps.is_available():
            self._device = "mps"
        elif torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"

        # Initialize text parser (always available)
        try:
            self.text_parser = TextEmotionParser()
        except Exception:
            self.text_parser = None

        # Load multimodal model if checkpoint exists
        checkpoint_path = Path("checkpoints/multimodal_emotion/multimodal_emotion.pt")
        if checkpoint_path.exists():
            try:
                self.multimodal_model = MultimodalEmotionModel(n_mels=64, embed_dim=256)
                ckpt = torch.load(checkpoint_path, map_location=self._device, weights_only=True)
                self.multimodal_model.load_state_dict(ckpt["model_state_dict"])
                self.multimodal_model = self.multimodal_model.to(self._device)
                self.multimodal_model.eval()
            except Exception:
                self.multimodal_model = None

    def _build_humanizer(self) -> DrumHumanizer:
        """Load DrumHumanizer with optional config from config/humanizer.json."""
        from pathlib import Path

        config_path = Path("config/humanizer.json")
        if config_path.exists():
            try:
                return DrumHumanizer(config_path=str(config_path))
            except Exception:
                return DrumHumanizer()
        return DrumHumanizer()

    def _build_emotion_keywords(self) -> Dict[str, Tuple[float, float, str]]:
        """Build emotion keyword to (valence, arousal, emotion) mapping."""
        return {
            # Negative valence, low arousal
            "grief": (-0.8, 0.3, "grief"),
            "sad": (-0.6, 0.3, "grief"),
            "loss": (-0.7, 0.3, "grief"),
            "mourning": (-0.8, 0.2, "grief"),
            "melancholy": (-0.5, 0.2, "grief"),
            "sorrow": (-0.7, 0.3, "grief"),

            # Negative valence, high arousal
            "anxiety": (-0.6, 0.8, "anxiety"),
            "anxious": (-0.6, 0.8, "anxiety"),
            "nervous": (-0.5, 0.7, "anxiety"),
            "panic": (-0.7, 0.9, "anxiety"),
            "fear": (-0.6, 0.8, "anxiety"),
            "worry": (-0.5, 0.6, "anxiety"),

            # Negative valence, high arousal (anger)
            "anger": (-0.7, 0.9, "anger"),
            "angry": (-0.7, 0.9, "anger"),
            "rage": (-0.9, 0.95, "anger"),
            "fury": (-0.8, 0.95, "anger"),
            "frustration": (-0.5, 0.7, "anger"),

            # Positive valence, low arousal
            "calm": (0.3, 0.2, "calm"),
            "peaceful": (0.4, 0.2, "calm"),
            "serene": (0.4, 0.15, "calm"),
            "relaxed": (0.3, 0.25, "calm"),
            "content": (0.4, 0.3, "calm"),

            # Positive valence, high arousal
            "hope": (0.6, 0.6, "hope"),
            "hopeful": (0.6, 0.6, "hope"),
            "joy": (0.8, 0.7, "hope"),
            "happy": (0.7, 0.6, "hope"),
            "excited": (0.7, 0.8, "hope"),
            "euphoria": (0.9, 0.9, "hope"),

            # Mixed/complex
            "nostalgia": (-0.2, 0.3, "nostalgia"),
            "nostalgic": (-0.2, 0.3, "nostalgia"),
            "bittersweet": (-0.1, 0.4, "nostalgia"),
            "wistful": (-0.2, 0.3, "nostalgia"),

            "tension": (-0.4, 0.7, "tension"),
            "suspense": (-0.4, 0.7, "tension"),
            "building": (-0.3, 0.6, "tension"),
            "uneasy": (-0.4, 0.6, "tension"),

            "catharsis": (0.2, 0.8, "catharsis"),
            "release": (0.3, 0.7, "catharsis"),
            "breakthrough": (0.4, 0.8, "catharsis"),

            "dissociation": (-0.3, 0.2, "dissociation"),
            "disconnected": (-0.3, 0.2, "dissociation"),
            "numb": (-0.4, 0.1, "dissociation"),
            "detached": (-0.3, 0.2, "dissociation"),

            "intimacy": (0.5, 0.3, "intimacy"),
            "intimate": (0.5, 0.3, "intimacy"),
            "vulnerable": (0.1, 0.4, "intimacy"),
            "tender": (0.5, 0.3, "intimacy"),
        }

    # ========== DECLARATIVE API ==========

    def generate_from_intent(
        self,
        intent: CompleteSongIntent
    ) -> GeneratedMusic:
        """
        Generate complete music from intent (simple, one-step).

        Args:
            intent: Complete song intent with all phases

        Returns:
            GeneratedMusic with all parameters
        """
        # Extract emotion from intent
        mood = intent.song_intent.mood_primary.lower() if intent.song_intent.mood_primary else "neutral"

        # Map mood to valence/arousal
        if mood in self._emotion_keywords:
            valence, arousal, emotion_key = self._emotion_keywords[mood]
        else:
            # Try to find partial match
            emotion_key = mood
            valence, arousal = -0.5, 0.4  # Default for unknown
            for keyword, (v, a, e) in self._emotion_keywords.items():
                if keyword in mood:
                    valence, arousal, emotion_key = v, a, e
                    break

        # Create emotional state
        emotional_state = EmotionalState(
            valence=Valence(int(valence * 2)),  # Convert to enum
            arousal=Arousal(int((arousal - 0.5) * 4)),  # Convert to enum
            primary_emotion=emotion_key
        )

        # Get musical parameters
        musical_params = get_parameters_for_state(emotional_state)

        # Override with technical constraints from intent
        if intent.technical_constraints.technical_tempo_range:
            tempo_min, tempo_max = intent.technical_constraints.technical_tempo_range
            musical_params.tempo_min = tempo_min
            musical_params.tempo_max = tempo_max
            musical_params.tempo_suggested = (tempo_min + tempo_max) // 2

        # Map to mixer parameters
        mixer_params = self.emotion_mapper.map_emotion_to_mixer(
            emotional_state,
            musical_params
        )

        production_preset = self.production_mapper.get_production_preset(
            emotion_key,
            genre=intent.technical_constraints.technical_genre or None,
            intensity=intent.song_intent.vulnerability_scale,
        )

        groove_settings = settings_from_intent(
            intent.song_intent.vulnerability_scale or "Medium",
            intent.technical_constraints.technical_groove_feel or "Organic/Breathing",
            mood_tension=intent.song_intent.mood_secondary_tension,
        )

        return GeneratedMusic(
            emotional_state=emotional_state,
            musical_params=musical_params,
            mixer_params=mixer_params,
            production_preset=production_preset,
            groove_settings=groove_settings,
            intent=intent
        )

    def generate_from_text(
        self,
        emotional_text: str
    ) -> GeneratedMusic:
        """
        Generate music from emotional text description.

        Uses neural emotion recognition when available, falls back to keyword matching.

        Args:
            emotional_text: Natural language emotional description

        Returns:
            GeneratedMusic with all parameters
        """
        import torch

        valence = 0.0
        arousal = 0.5
        primary_emotion = "neutral"
        modifiers: List[str] = []

        # Try neural approach first
        if self.use_neural and self.text_parser and self.multimodal_model:
            try:
                parsed = self.text_parser.parse(emotional_text)
                text_feat = prepare_text_features(parsed).unsqueeze(0).to(self._device)

                with torch.no_grad():
                    outputs = self.multimodal_model(text_features=text_feat)

                BASE_EMOTIONS = ["HAPPY", "SAD", "ANGRY", "FEAR", "SURPRISE", "DISGUST"]
                pred_idx = outputs["base_emotion"].argmax(dim=-1).item()

                valence = outputs["valence"][0].item()
                arousal = outputs["arousal"][0].item()
                primary_emotion = self._neural_emotion_to_preset(BASE_EMOTIONS[pred_idx])
                modifiers = parsed.modifiers

            except Exception:
                # Fall back to keyword matching
                pass

        # Keyword fallback if neural didn't work
        if primary_emotion == "neutral":
            text_lower = emotional_text.lower()
            for keyword, (v, a, emotion) in self._emotion_keywords.items():
                if keyword in text_lower:
                    valence, arousal = v, a
                    primary_emotion = emotion
                    break

        # Create emotional state
        emotional_state = EmotionalState(
            valence=Valence(max(-2, min(2, int(valence * 2)))),
            arousal=Arousal(max(-2, min(2, int((arousal - 0.5) * 4)))),
            primary_emotion=primary_emotion,
            has_intrusions="ptsd_intrusion" in modifiers,
        )

        # Get musical parameters
        musical_params = get_parameters_for_state(emotional_state)

        # Map to mixer parameters
        mixer_params = self.emotion_mapper.map_emotion_to_mixer(
            emotional_state,
            musical_params
        )

        # Determine intensity from arousal
        if arousal > 0.7:
            intensity = "high"
        elif arousal < 0.4:
            intensity = "low"
        else:
            intensity = "Medium"

        production_preset = self.production_mapper.get_production_preset(
            primary_emotion,
            intensity=intensity,
        )

        groove_settings = settings_from_intent(
            intensity.capitalize() if intensity != "Medium" else "Medium",
            "Organic/Breathing",
            mood_tension=0.5 + (1 - valence) * 0.25,  # More tension for negative valence
        )

        return GeneratedMusic(
            emotional_state=emotional_state,
            musical_params=musical_params,
            mixer_params=mixer_params,
            production_preset=production_preset,
            groove_settings=groove_settings,
        )

    def _neural_emotion_to_preset(self, neural_emotion: str) -> str:
        """Map neural model emotion to production preset name."""
        mapping = {
            "HAPPY": "hope",
            "SAD": "grief",
            "ANGRY": "anger",
            "FEAR": "anxiety",
            "SURPRISE": "hope",
            "DISGUST": "tension",
        }
        return mapping.get(neural_emotion, "neutral")

    def export_to_logic(
        self,
        music: GeneratedMusic,
        output_base: str
    ) -> Dict[str, str]:
        """
        Export to Logic Pro format.

        Args:
            music: Generated music
            output_base: Base filename (without extension)

        Returns:
            Dict with paths to created files
        """
        output_base = Path(output_base).stem

        # Export mixer automation
        automation_path = f"{output_base}_automation.json"
        export_to_logic_automation(music.mixer_params, automation_path)
        music.automation_path = automation_path

        result = {
            "automation": automation_path,
        }

        return result

    def create_intent(
        self,
        title: str,
        core_event: str,
        mood_primary: str,
        technical_key: str = "C",
        technical_mode: str = "major",
        tempo_range: Tuple[int, int] = (80, 120),
        rule_to_break: str = "",
        rule_justification: str = "",
        **kwargs: Any,
    ) -> CompleteSongIntent:
        """
        Create a CompleteSongIntent from parameters.

        Args:
            title: Song title
            core_event: The inciting moment/realization
            mood_primary: Primary emotion
            technical_key: Musical key
            technical_mode: Mode (major, minor, etc.)
            tempo_range: (min, max) BPM
            rule_to_break: Optional rule to break
            rule_justification: Why break the rule
            **kwargs: Additional intent fields

        Returns:
            CompleteSongIntent
        """
        return CompleteSongIntent(
            title=title,
            song_root=SongRoot(
                core_event=core_event,
                core_resistance=kwargs.get("core_resistance", ""),
                core_longing=kwargs.get("core_longing", ""),
                core_stakes=kwargs.get("core_stakes", ""),
                core_transformation=kwargs.get("core_transformation", ""),
            ),
            song_intent=SongIntent(
                mood_primary=mood_primary,
                mood_secondary_tension=kwargs.get("mood_secondary_tension", 0.5),
                imagery_texture=kwargs.get("imagery_texture", ""),
                vulnerability_scale=kwargs.get("vulnerability_scale", "Medium"),
                narrative_arc=kwargs.get("narrative_arc", ""),
            ),
            technical_constraints=TechnicalConstraints(
                technical_genre=kwargs.get("technical_genre", ""),
                technical_tempo_range=tempo_range,
                technical_key=technical_key,
                technical_mode=technical_mode,
                technical_groove_feel=kwargs.get("technical_groove_feel", ""),
                technical_rule_to_break=rule_to_break,
                rule_breaking_justification=rule_justification,
            ),
            system_directive=SystemDirective(
                output_target=kwargs.get("output_target", ""),
                output_feedback_loop=kwargs.get("output_feedback_loop", ""),
            ),
        )

    def suggest_rules(self, emotion: str) -> List[Dict]:
        """
        Get rule-breaking suggestions for an emotion.

        Args:
            emotion: Target emotion

        Returns:
            List of rule-breaking suggestions
        """
        return suggest_rule_break(emotion)

    # ========== ADVANCED FEATURES ==========

    def get_suggestions(self, context: Dict[str, Any]) -> List[Dict]:
        """
        Get AI-powered suggestions based on current context.

        Args:
            context: Dict with current musical/emotional state

        Returns:
            List of suggestion dicts with action, reason, priority
        """
        if HAS_SUGGESTIONS:
            engine = SuggestionEngine()
            suggestions = engine.generate_suggestions(context)
            return [
                {
                    "type": s.suggestion_type.value,
                    "title": s.title,
                    "description": s.description,
                    "action": s.action,
                    "confidence": s.confidence,
                    "explanation": s.explanation,
                }
                for s in suggestions
            ]
        return []

    def generate_arrangement(
        self,
        music: "GeneratedMusic",
        structure: List[str],
    ) -> Dict[str, Any]:
        """
        Generate full arrangement from music parameters.

        Args:
            music: Generated music with emotion/params
            structure: List of section names

        Returns:
            Arrangement dict with per-section instrument patterns
        """
        if HAS_ARRANGEMENT:
            generator = ArrangementGenerator()
            return generator.generate(
                emotion=music.emotional_state.primary_emotion,
                tempo=music.musical_params.tempo_suggested,
                structure=structure,
            )
        return {"error": "Arrangement generator not available"}

    def visualize_emotion_trajectory(
        self,
        emotions: List[Dict[str, float]],
        output_path: Optional[str] = None,
    ) -> Any:
        """
        Create a Spectocloud visualization of emotion trajectory.

        Args:
            emotions: List of dicts with valence, arousal, intensity over time
            output_path: Optional path to save visualization

        Returns:
            Visualization data or image path
        """
        if HAS_SPECTOCLOUD:
            viz = Spectocloud()
            # Convert emotion trajectory to MIDI-like events for visualization
            midi_events = []
            for i, emo in enumerate(emotions):
                midi_events.append({
                    "type": "note_on",
                    "time": i * 0.5,  # 0.5s per emotion point
                    "note": int(60 + emo.get("valence", 0) * 12),
                    "velocity": int(64 + emo.get("arousal", 0.5) * 50),
                })
            viz.process_midi(midi_events, len(emotions) * 0.5, emotion_trajectory=emotions)
            if output_path:
                viz.render_static_frame(len(viz.frames) // 2, output_path=output_path, show=False)
                return {"path": output_path, "frames": len(viz.frames)}
            return {"frames": len(viz.frames), "data": viz.export_data if hasattr(viz, "export_data") else None}
        return {"error": "Spectocloud not available"}

    def get_dynamics_profile(
        self,
        sections: List[str],
        section_bars: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Get guide-based dynamics profile for song structure.

        Args:
            sections: List of section names (intro, verse, chorus, etc.)
            section_bars: Optional bars per section

        Returns:
            Complete dynamics profile with automation curve
        """
        from music_brain.production.dynamics_engine import SongStructure

        structure = SongStructure(sections=sections, section_bars=section_bars)
        profile = self.dynamics_engine.get_arrangement_profile(structure)

        return {
            "sections": {
                name: {
                    "level": dyn.level,
                    "db": dyn.target_db,
                    "density": dyn.density,
                    "drums": dyn.drums,
                    "bass": dyn.bass,
                    "guitar": dyn.guitar,
                    "notes": dyn.notes,
                }
                for name, dyn in profile.sections.items()
            },
            "automation": profile.automation.points,
            "peak_section": profile.peak_section,
            "quietest_section": profile.quietest_section,
            "contrast_ratio": profile.contrast_ratio,
            "suggestions": self.dynamics_engine.suggest_contrast_improvements(profile),
        }

    def humanize_drums(
        self,
        events: List[Dict[str, Any]],
        style: str = "standard",
        ppq: int = 480,
    ) -> List[Dict[str, Any]]:
        """
        Apply guide-based humanization to drum events.

        Args:
            events: MIDI-like event dicts
            style: Style preset (hip-hop, rock, jazzy, edm, lofi, etc.)
            ppq: Pulses per quarter note

        Returns:
            Humanized event list
        """
        return self.drum_humanizer.apply_guide_rules(events, style=style, ppq=ppq)

    def list_humanizer_styles(self) -> List[str]:
        """Get available drum humanizer style presets."""
        return list(self.drum_humanizer._style_presets.keys())

    def get_capabilities(self) -> Dict[str, bool]:
        """Get available API capabilities."""
        return {
            "neural_emotion": self.use_neural and self.multimodal_model is not None,
            "text_parser": self.text_parser is not None,
            "spectocloud": HAS_SPECTOCLOUD,
            "suggestions": HAS_SUGGESTIONS,
            "arrangement": HAS_ARRANGEMENT,
            "dynamics_engine": True,
            "drum_humanizer": True,
            "device": self._device,
        }

    # ========== FLUENT API ==========

    def process(self, emotional_text: str) -> 'FluentChain':
        """Start a fluent processing chain."""
        return FluentChain(emotional_text, self)


class FluentChain:
    """
    Fluent API for step-by-step control over emotion-to-music mapping.

    Usage:
        brain = MusicBrain()
        result = (brain.process("grief and loss")
                       .map_to_emotion()
                       .map_to_music()
                       .map_to_mixer()
                       .export_logic("output_automation.json"))
    """

    def __init__(self, emotional_text: str, brain: MusicBrain):
        self.brain = brain
        self.emotional_text = emotional_text
        self.emotional_state: Optional[EmotionalState] = None
        self.musical_params: Optional[MusicalParameters] = None
        self.mixer_params: Optional[MixerParameters] = None
        self._paths: Dict[str, str] = {}

    def map_to_emotion(self) -> 'FluentChain':
        """Map text to emotional state."""
        text_lower = self.emotional_text.lower()

        valence = 0.0
        arousal = 0.5
        primary_emotion = "neutral"

        for keyword, (v, a, emotion) in self.brain._emotion_keywords.items():
            if keyword in text_lower:
                valence, arousal = v, a
                primary_emotion = emotion
                break

        self.emotional_state = EmotionalState(
            valence=Valence(max(-2, min(2, int(valence * 2)))),
            arousal=Arousal(max(-2, min(2, int((arousal - 0.5) * 4)))),
            primary_emotion=primary_emotion
        )
        return self

    def map_to_music(self) -> 'FluentChain':
        """Map emotion to musical parameters."""
        if not self.emotional_state:
            self.map_to_emotion()
        assert self.emotional_state is not None
        self.musical_params = get_parameters_for_state(self.emotional_state)
        return self

    def map_to_mixer(self) -> 'FluentChain':
        """Map to mixer parameters."""
        if not self.musical_params:
            self.map_to_music()
        assert self.emotional_state is not None
        assert self.musical_params is not None
        self.mixer_params = self.brain.emotion_mapper.map_emotion_to_mixer(
            self.emotional_state,
            self.musical_params
        )
        return self

    def with_tempo(self, tempo: int) -> 'FluentChain':
        """Override tempo."""
        if not self.musical_params:
            self.map_to_music()
        assert self.musical_params is not None
        self.musical_params.tempo_suggested = tempo
        return self

    def with_dissonance(self, dissonance: float) -> 'FluentChain':
        """Override dissonance level (0.0-1.0)."""
        if not self.musical_params:
            self.map_to_music()
        assert self.musical_params is not None
        self.musical_params.dissonance = max(0.0, min(1.0, dissonance))
        return self

    def with_timing(self, feel: str) -> 'FluentChain':
        """Override timing feel (ahead, on, behind)."""
        if not self.musical_params:
            self.map_to_music()
        assert self.musical_params is not None
        feel_map = {
            "ahead": TimingFeel.AHEAD,
            "on": TimingFeel.ON,
            "behind": TimingFeel.BEHIND,
        }
        if feel.lower() in feel_map:
            self.musical_params.timing_feel = feel_map[feel.lower()]
        return self

    def export_logic(self, output_path: str) -> Dict[str, str]:
        """Export to Logic Pro and return paths."""
        if not self.mixer_params:
            self.map_to_mixer()
        assert self.mixer_params is not None

        automation_path = export_to_logic_automation(
            self.mixer_params,
            output_path
        )
        self._paths["automation"] = automation_path

        return {
            "automation": automation_path,
            "emotional_state": str(self.emotional_state.primary_emotion) if self.emotional_state else "neutral",
            "tempo": str(self.musical_params.tempo_suggested) if self.musical_params else "unknown",
        }

    def export_json(self, output_path: str) -> str:
        """Export full settings to JSON."""
        if not self.mixer_params:
            self.map_to_mixer()
        assert self.mixer_params is not None

        export_mixer_settings(self.mixer_params, output_path, format="json")
        self._paths["json"] = output_path
        return output_path

    def get(self) -> Dict[str, Any]:
        """Get current state."""
        return {
            "emotional_text": self.emotional_text,
            "emotional_state": {
                "primary_emotion": self.emotional_state.primary_emotion if self.emotional_state else None,
                "valence": self.emotional_state.valence if self.emotional_state else None,
                "arousal": self.emotional_state.arousal if self.emotional_state else None,
            } if self.emotional_state else None,
            "musical_params": {
                "tempo": self.musical_params.tempo_suggested if self.musical_params else None,
                "dissonance": self.musical_params.dissonance if self.musical_params else None,
                "timing_feel": self.musical_params.timing_feel.value if self.musical_params else None,
            } if self.musical_params else None,
            "mixer_params": self.mixer_params.to_dict() if self.mixer_params else None,
            "paths": self._paths,
        }

    def describe(self) -> str:
        """Get human-readable description of current state."""
        lines = []

        if self.emotional_state:
            lines.extend([
                f"Emotion: {self.emotional_state.primary_emotion}",
                f"Valence: {self.emotional_state.valence}",
                f"Arousal: {self.emotional_state.arousal}",
            ])

        if self.musical_params:
            lines.extend([
                "",
                f"Tempo: {self.musical_params.tempo_suggested} BPM",
                f"Timing: {self.musical_params.timing_feel.value}",
                f"Dissonance: {self.musical_params.dissonance:.0%}",
            ])

        if self.mixer_params:
            lines.extend([
                "",
                f"Mixer: {self.mixer_params.description}",
                f"Reverb: {self.mixer_params.reverb_mix:.0%}",
                f"Compression: {self.mixer_params.compression_ratio:.1f}:1",
            ])

        return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_generate(emotional_text: str) -> GeneratedMusic:
    """Quick helper to generate music from text."""
    brain = MusicBrain()
    return brain.generate_from_text(emotional_text)


def quick_export(emotional_text: str, output_path: str) -> Dict[str, str]:
    """Quick helper to generate and export to Logic Pro."""
    brain = MusicBrain()
    music = brain.generate_from_text(emotional_text)
    return brain.export_to_logic(music, output_path)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MUSIC BRAIN EMOTION API - EXAMPLES")
    print("=" * 70)

    brain = MusicBrain()

    # Example 1: Simple text-based generation
    print("\n[Example 1: Text -> Music]")
    music = brain.generate_from_text("grief and loss processing")
    print(f"  Emotion: {music.emotional_state.primary_emotion}")
    print(f"  Tempo: {music.musical_params.tempo_suggested} BPM")
    print(f"  Reverb: {music.mixer_params.reverb_mix:.0%}")
    print(f"  Description: {music.mixer_params.description}")

    # Example 2: Fluent API
    print("\n[Example 2: Fluent Chain]")
    result = (brain.process("anxiety and tension")
                   .map_to_emotion()
                   .map_to_music()
                   .with_tempo(110)
                   .map_to_mixer()
                   .get())
    print(f"  Emotion: {result['emotional_state']['primary_emotion']}")
    print(f"  Tempo: {result['musical_params']['tempo']}")
    print(f"  Compression: {result['mixer_params']['compression']['ratio']:.1f}:1")

    # Example 3: From intent
    print("\n[Example 3: From Intent]")
    intent = brain.create_intent(
        title="Test Song",
        core_event="Processing loss",
        mood_primary="grief",
        technical_key="F",
        technical_mode="major",
        tempo_range=(78, 86),
        rule_to_break="HARMONY_ModalInterchange",
        rule_justification="Bbm creates bittersweet hope"
    )
    music_from_intent = brain.generate_from_intent(intent)
    print(f"  Title: {intent.title}")
    print(f"  Emotion: {music_from_intent.emotional_state.primary_emotion}")
    print(f"  Tempo: {music_from_intent.musical_params.tempo_suggested} BPM")

    # Example 4: Export
    print("\n[Example 4: Export to Logic Pro]")
    paths = brain.export_to_logic(music, "example_grief")
    print(f"  Automation file: {paths['automation']}")

    # List available presets
    print("\n[Available Mixer Presets]")
    print(f"  {', '.join(brain.emotion_mapper.list_presets())}")

    print("\n" + "=" * 70)
