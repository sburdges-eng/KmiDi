"""
Map emotional intent to production presets sourced from the production guides.

Goals / TODOs:
- Ingest Production_Workflows markdown into structured presets.
- Add genre-aware overrides (hip-hop, rock, jazz, EDM, acoustic, etc.).
- Align outputs with groove humanizer + dynamics engines for end-to-end flow.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ProductionPreset:
    """Lightweight container for guide-driven production choices."""

    drum_style: str
    dynamics_level: str
    arrangement_density: float
    groove_template: Optional[str] = None
    genre: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        """Serialize to a plain dictionary."""
        return {
            "drum_style": self.drum_style,
            "dynamics_level": self.dynamics_level,
            "arrangement_density": self.arrangement_density,
            "groove_template": self.groove_template,
            "genre": self.genre,
            "notes": list(self.notes),
        }


class EmotionProductionMapper:
    """
    Convert emotion/genre hints into production presets.

    This is the code counterpart to the production guides (dynamics, drum
    programming, groove, and humanization). It provides a stable API the rest
    of the system can call without needing to parse markdown directly.
    """

    def __init__(self) -> None:
        self._mood_presets: Dict[str, ProductionPreset] = {
            "calm": ProductionPreset(
                drum_style="minimal",
                dynamics_level="mp",
                arrangement_density=0.35,
                groove_template="lofi",
                notes=[
                    "Sparse drums; gentle swing (Groove and Rhythm Guide).",
                    "Dynamics/Arrangement: leave space, emphasize texture.",
                ],
            ),
            "uplifting": ProductionPreset(
                drum_style="punchy_pop",
                dynamics_level="f",
                arrangement_density=0.65,
                groove_template="pop",
                notes=[
                    "Clear downbeats; supportive bass (Bass Guide).",
                    "Chorus lifts with brighter EQ choices.",
                ],
            ),
            "aggressive": ProductionPreset(
                drum_style="heavy",
                dynamics_level="ff",
                arrangement_density=0.8,
                groove_template="rock",
                notes=[
                    "Tight timing, low vulnerability (Drum Guide).",
                    "Compression/EQ: emphasize punch and midrange.",
                ],
            ),
            "dark": ProductionPreset(
                drum_style="moody",
                dynamics_level="mf",
                arrangement_density=0.55,
                groove_template="trip-hop",
                notes=[
                    "Behind-the-beat; filtered drums; leave ambience space.",
                    "Ambient/Atmospheric: long tails, evolving layers.",
                ],
            ),
            "nostalgic": ProductionPreset(
                drum_style="lofi",
                dynamics_level="mp",
                arrangement_density=0.45,
                groove_template="soul",
                notes=[
                    "Swinged hats, softened transients, tape saturation.",
                    "Lo-Fi Production Guide: gentle reverb/delay glue.",
                ],
            ),
            "neutral": ProductionPreset(
                drum_style="standard",
                dynamics_level="mf",
                arrangement_density=0.5,
                groove_template="neutral",
                notes=[
                    "Baseline preset; adjust via intensity and genre inputs.",
                ],
            ),
        }

    def get_production_preset(
        self,
        emotion: str,
        genre: Optional[str] = None,
        intensity: Optional[str] = None,
    ) -> ProductionPreset:
        """
        Return a preset derived from the emotion and optional genre/intensity.

        Args:
            emotion: Free-form emotion or intent label.
            genre: Optional genre hint to bias template selection.
            intensity: Optional label such as "low", "medium", "high".
        """
        mood = self._classify_emotion(emotion)
        base = self._mood_presets.get(mood, self._mood_presets["neutral"])
        preset = ProductionPreset(
            drum_style=base.drum_style,
            dynamics_level=base.dynamics_level,
            arrangement_density=base.arrangement_density,
            groove_template=base.groove_template,
            genre=genre,
            # Copy to avoid mutating the internal template when callers append.
            notes=list(base.notes) if base.notes else [],
        )

        if intensity:
            preset.arrangement_density = self._apply_intensity(
                preset.arrangement_density, intensity
            )
            preset.dynamics_level = self._scale_dynamics(
                preset.dynamics_level, intensity
            )

        if genre:
            preset.notes = [f"Genre-biased preset for {genre}."] + preset.notes

        return preset

    def get_drum_style(self, emotion: str) -> str:
        """Shortcut to only fetch drum style."""
        return self.get_production_preset(emotion).drum_style

    def get_dynamics_level(
        self,
        emotion: str,
        intensity: Optional[str] = None,
    ) -> str:
        """Shortcut to only fetch dynamics level."""
        preset = self.get_production_preset(emotion, intensity=intensity)
        return preset.dynamics_level

    def describe_sources(self) -> List[str]:
        """
        List guide files that should inform this mapper.

        This keeps the ingestion list close to the code until we wire in an
        automated parser for the Production_Workflows markdown set.
        """
        return [
            "Production_Workflows/Bass Programming Guide.md",
            "Production_Workflows/Drum Programming Guide.md",
            "Production_Workflows/Dynamics and Arrangement Guide.md",
            "Production_Workflows/Humanization Cheat Sheet.md",
            "Production_Workflows/Compression Deep Dive Guide.md",
            "Production_Workflows/EQ Deep Dive Guide.md",
            "Production_Workflows/Electronic EDM Production Guide.md",
            "Production_Workflows/Groove and Rhythm Guide.md",
            "Production_Workflows/Humanizing Your Music.md",
        ]

    def _classify_emotion(self, emotion: str) -> str:
        if not emotion:
            return "neutral"

        text = emotion.lower()
        if any(
            word in text
            for word in ["calm", "peace", "relax", "gentle", "soft"]
        ):
            return "calm"
        if any(word in text for word in ["happy", "uplift", "hope", "bright"]):
            return "uplifting"
        if any(
            word in text
            for word in ["angry", "rage", "aggressive", "drive"]
        ):
            return "aggressive"
        if any(
            word in text
            for word in ["dark", "brood", "moody", "tension"]
        ):
            return "dark"
        if any(
            word in text
            for word in ["nostalgia", "nostalgic", "memory", "retro"]
        ):
            return "nostalgic"
        return "neutral"

    def _apply_intensity(self, density: float, intensity: str) -> float:
        intensity_text = intensity.lower()
        if any(
            word in intensity_text
            for word in ["high", "big", "strong", "up"]
        ):
            return self._clamp(density + 0.15)
        if any(
            word in intensity_text
            for word in ["low", "quiet", "intimate", "down"]
        ):
            return self._clamp(density - 0.15)
        return self._clamp(density)

    def _scale_dynamics(self, level: str, intensity: str) -> str:
        order = ["pp", "p", "mp", "mf", "f", "ff"]
        idx = order.index(level) if level in order else order.index("mf")

        text = intensity.lower()
        if any(word in text for word in ["high", "big", "strong", "up"]):
            idx = min(len(order) - 1, idx + 1)
        if any(word in text for word in ["low", "quiet", "intimate", "down"]):
            idx = max(0, idx - 1)
        return order[idx]

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, value))
