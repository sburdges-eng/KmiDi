"""
LocalMultiModelOrchestrator
---------------------------
Coordinates local-only components with strict separation of concerns:
 - BrainController (Mistral 7B via llama.cpp) -> planning
 - MidiGenerationPipeline -> MIDI generation
 - ImageGenerationJob -> optional images
 - AudioTextureJob -> optional textures (experimental)
 - VoiceGenerationPipeline -> optional vocals (clone vs performer)

Execution model:
 - Sequential by default
 - Explicit concurrency only if marked safe (not enabled by default)
 - Explicit load/unload for heavy models to respect 16 GB memory
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List
import traceback
import time
import os

from music_brain.intelligence.brain_controller import (
    BrainController,
    BrainConfig,
)
from music_brain.tier1.midi_pipeline import MidiGenerationPipeline, MidiGenerationResult
from music_brain.generators.image_generator import ImageGenerationJob, ImageGenerationConfig
from music_brain.generators.audio_texture import AudioTextureJob, AudioTextureConfig
from music_brain.tier1.voice_pipeline import (
    VoiceGenerationPipeline,
    VoiceIdentity,
    VocalNote,
    VocalExpression,
)
from music_brain.session.intent_schema import CompleteSongIntent


@dataclass
class OrchestrationConfig:
    # Brain / LLM
    mistral_gguf_path: Path
    mistral_ctx: int = 4096
    mistral_seed: Optional[int] = None
    llama_threads: Optional[int] = None
    mistral_gpu_layers: Optional[int] = None
    keep_brain_loaded: bool = False

    # Generation toggles
    enable_images: bool = True
    enable_audio_texture: bool = False
    enable_voice: bool = False

    # Paths
    output_root: Path = Path.home() / "Music" / "iDAW_Output"

    # Determinism
    midi_seed: int = 17
    midi_length: int = 16
    midi_device: str = "auto"  # allow "mps" on Apple Silicon


@dataclass
class OrchestrationResult:
    intent: CompleteSongIntent
    midi: Optional[MidiGenerationResult] = None
    image_paths: List[Path] = field(default_factory=list)
    audio_textures: List[Path] = field(default_factory=list)
    voice_outputs: List[Path] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class LocalMultiModelOrchestrator:
    def __init__(self, config: OrchestrationConfig):
        self.config = config
        self._brain = self._create_brain()
        self._brain_loaded = False
        self._midi = MidiGenerationPipeline(
            seed=self.config.midi_seed, device=self.config.midi_device
        )
        self._image_job = (
            ImageGenerationJob(ImageGenerationConfig(output_dir=self.config.output_root / "images"))
            if config.enable_images
            else None
        )
        self._audio_texture = (
            AudioTextureJob(AudioTextureConfig(output_dir=self.config.output_root / "audio_textures"))
            if config.enable_audio_texture
            else None
        )
        self._voice = (
            VoiceGenerationPipeline()
            if config.enable_voice
            else None
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def execute(self, user_intent: str) -> OrchestrationResult:
        """
        Full pipeline: parse intent then run generators.
        """
        intent = self._run_brain(user_intent)
        return self.execute_with_intent(intent)

    def execute_with_intent(self, intent: CompleteSongIntent) -> OrchestrationResult:
        result = OrchestrationResult(intent=intent)
        try:
            midi_result = self._midi.generate(intent, length=self.config.midi_length)
            result.midi = midi_result

            # Optional branches
            if self._image_job:
                prompts = self._safe_expand_prompts(intent)
                try:
                    images = self._image_job.generate(
                        prompt=prompts.get("image_prompt", ""),
                        style_rules=self._parse_json_like(prompts.get("image_style_rules", "{}")),
                        num_images=1,
                    )
                    result.image_paths.extend(images)
                except Exception:
                    result.errors.append("Image generation failed")

            if self._audio_texture:
                prompts = self._safe_expand_prompts(intent)
                try:
                    audio_path = self._audio_texture.generate_texture(
                        prompt=prompts.get("audio_texture_prompt", "texture"),
                        duration_seconds=8.0,
                    )
                    result.audio_textures.append(audio_path)
                except Exception:
                    result.errors.append("Audio texture generation failed")

            if self._voice:
                # Placeholder: voice rendering requires notes & identity
                try:
                    identity = VoiceIdentity(name="default")
                    notes = [
                        VocalNote(pitch=60, start_beat=0.0, duration_beats=1.0, lyric="la", expression={"intensity": 0.6}),
                        VocalNote(pitch=62, start_beat=1.0, duration_beats=1.0, lyric="la", expression={"intensity": 0.6}),
                    ]
                    expr = VocalExpression(intensity_curve=[0.6, 0.6], phrasing_offsets=[0.0, 0.0])
                    voice_out = self._voice.generate(identity=identity, notes=notes, expression=expr)
                    if voice_out.audio_path:
                        result.voice_outputs.append(voice_out.audio_path)
                except Exception:
                    result.errors.append("Voice generation failed")

        except Exception:
            result.errors.append(traceback.format_exc())

        return result

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _create_brain(self) -> BrainController:
        default_threads = max(1, os.cpu_count() or 1)
        default_gpu_layers = 35  # tuned for 7B; override via config
        brain_cfg = BrainConfig(
            model_path=self.config.mistral_gguf_path,
            n_ctx=self.config.mistral_ctx,
            seed=self.config.mistral_seed,
            n_threads=self.config.llama_threads or default_threads,
            n_gpu_layers=(
                self.config.mistral_gpu_layers
                if self.config.mistral_gpu_layers is not None
                else default_gpu_layers
            ),
        )
        return BrainController(brain_cfg)

    def _run_brain(self, user_intent: str) -> CompleteSongIntent:
        self._ensure_brain_loaded()
        try:
            intent = self._brain.parse_intent(user_intent)
        finally:
            self._maybe_unload_brain()
        return intent

    def _safe_expand_prompts(self, intent: CompleteSongIntent) -> Dict[str, str]:
        try:
            self._ensure_brain_loaded()
            return self._brain.expand_prompts(intent)
        except Exception:
            return {}
        finally:
            self._maybe_unload_brain()

    @staticmethod
    def _parse_json_like(text: str) -> Dict[str, Any]:
        import json

        try:
            return json.loads(text)
        except Exception:
            return {}

    # ------------------------------------------------------------------ #
    # Brain lifecycle helpers
    # ------------------------------------------------------------------ #
    def _ensure_brain_loaded(self):
        if not self._brain_loaded:
            self._brain.load()
            self._brain_loaded = True

    def _maybe_unload_brain(self):
        if not self.config.keep_brain_loaded and self._brain_loaded:
            self._brain.unload()
            self._brain_loaded = False

    # ------------------------------------------------------------------ #
    # Output helpers
    # ------------------------------------------------------------------ #
    def export_midi(
        self,
        midi_result: MidiGenerationResult,
        *,
        output_dir: Optional[Path] = None,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Export a MidiGenerationResult to disk. Keeps everything local/deterministic.
        """
        target_dir = output_dir or (self.config.output_root / "midi")
        target_dir.mkdir(parents=True, exist_ok=True)
        safe_name = filename or f"kmidi_{int(time.time())}.mid"
        return self._midi.export_midi(midi_result, target_dir / safe_name)


# Convenience factory
def create_local_orchestrator(
    mistral_gguf_path: str,
    enable_images: bool = True,
    enable_audio_texture: bool = False,
    enable_voice: bool = False,
) -> LocalMultiModelOrchestrator:
    cfg = OrchestrationConfig(
        mistral_gguf_path=Path(mistral_gguf_path),
        enable_images=enable_images,
        enable_audio_texture=enable_audio_texture,
        enable_voice=enable_voice,
    )
    return LocalMultiModelOrchestrator(cfg)

