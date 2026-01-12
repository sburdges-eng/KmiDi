"""
Orchestration Layer - Local Control Loop.

Coordinates Mistral inference, KmiDi MIDI generation, image diffusion jobs,
and optional audio diffusion jobs.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict

from music_brain.intelligence.brain_controller import BrainController, BrainConfig
from music_brain.orchestration.kmidi_generator import KmiDiGenerator
from music_brain.session.intent_schema import CompleteSongIntent
from music_brain.orchestration.stable_diffusion_generator import StableDiffusionGenerator
from music_brain.orchestration.audio_diffusion_generator import AudioDiffusionGenerator

logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Orchestration layer to coordinate AI music and image generation.
    """

    def __init__(
        self,
        mistral_model_path: Path,
        kmidi_model_registry_path: Path,
        stable_diffusion_model_path: Path,
        audio_diffusion_model_path: Path,
    ):
        self.brain_controller = BrainController(
            config=BrainConfig(model_path=mistral_model_path)
        )
        self.kmidi_generator = KmiDiGenerator(model_registry_path=kmidi_model_registry_path)
        self.stable_diffusion_generator = StableDiffusionGenerator(model_path=stable_diffusion_model_path)
        self.audio_diffusion_generator = AudioDiffusionGenerator(model_path=audio_diffusion_model_path)

    async def initialize(self):
        """
        Initialize and load all necessary models.
        """
        logger.info("Initializing Orchestrator: loading models...")
        self.brain_controller.load()
        await self.stable_diffusion_generator._load_model()
        await self.audio_diffusion_generator._load_model()
        logger.info("Orchestrator initialized.")

    async def shutdown(self):
        """
        Release all loaded model resources.
        """
        logger.info("Shutting down Orchestrator: unloading models...")
        self.brain_controller.unload()
        await self.stable_diffusion_generator._unload_model()
        await self.audio_diffusion_generator._unload_model()
        logger.info("Orchestrator shut down.")

    async def process_user_intent(self, user_text: str) -> Dict[str, Any]:
        """
        Main entry point for processing user intent.

        Args:
            user_text: Natural language user request.

        Returns:
            A dictionary containing generated MIDI plans, image prompts, etc.
        """
        logger.info(f"Processing user intent: \"{user_text}\"")

        # 1. Parse user intent using Mistral
        structured_intent: CompleteSongIntent = await self._run_with_resource_check(
            self.brain_controller.parse_intent, user_text, required_ram_gb=1.5 # Mistral 7B (Q4_K_M) approx 4.3GB model, but inference might be less
        )
        logger.debug(f"Parsed intent: {structured_intent.to_dict()}")

        # 2. Expand prompts for downstream generators
        expanded_prompts = await self._run_with_resource_check(
            self.brain_controller.expand_prompts, structured_intent, required_ram_gb=0.5 # Minimal additional RAM
        )
        logger.debug(f"Expanded prompts: {expanded_prompts}")

        # 3. Generate MIDI plan
        midi_plan = await self.kmidi_generator.generate_midi_plan(structured_intent)
        logger.debug(f"Generated MIDI plan: {midi_plan}")

        # 4. Generate image prompts + style constraints (placeholder)
        generated_image_path = await self._run_with_resource_check(
            self.stable_diffusion_generator.generate_image,
            image_prompt=expanded_prompts.get("image_prompt", ""),
            style_rules=json.loads(expanded_prompts.get("image_style_rules", "{}")),
            required_ram_gb=4.0 # Stable Diffusion 1.5 (fp16) requires ~4GB VRAM
        )

        image_output = {
            "image_prompt": expanded_prompts.get("image_prompt", ""),
            "image_style_rules": expanded_prompts.get("image_style_rules", "{}"),
            "generated_image_path": str(generated_image_path),
        }

        # 5. Generate audio texture prompts
        generated_audio_path = await self._run_with_resource_check(
            self.audio_diffusion_generator.generate_audio,
            prompt=expanded_prompts.get("audio_texture_prompt", ""),
            required_ram_gb=2.0 # MusicGen small/medium might need around 2-4GB RAM
        )

        audio_texture_output = {
            "audio_texture_prompt": expanded_prompts.get("audio_texture_prompt", ""),
            "generated_audio_path": str(generated_audio_path),
        }

        # 6. Provide explanations (example)
        explanation_midi = self.brain_controller.explain_decision(
            structured_intent, "chord progression in verse"
        )
        logger.debug(f"Explanation for MIDI: {explanation_midi}")

        return {
            "structured_intent": structured_intent.to_dict(),
            "expanded_prompts": expanded_prompts,
            "midi_plan": midi_plan,
            "image_output": image_output,
            "audio_texture_output": audio_texture_output,
            "explanations": {
                "midi_decision": explanation_midi
            },
        }

    async def run_midi_generation_job(self, midi_plan: Dict[str, Any]) -> Path:
        """
        Runs a standalone MIDI generation job.
        """
        logger.info("Running MIDI generation job.")
        midi_file_path = await self.kmidi_generator.generate_midi_file(midi_plan)
        logger.info(f"MIDI file generated: {midi_file_path}")
        return midi_file_path

    async def run_image_diffusion_job(self, image_prompt: str, style_rules: str) -> Path:
        """
        Runs an image diffusion job (placeholder).
        """
        logger.info(f"Running image diffusion job for prompt: {image_prompt}")
        output_image_path = await self._run_with_resource_check(
            self.stable_diffusion_generator.generate_image,
            prompt=image_prompt,
            style_rules=json.loads(style_rules),
            required_ram_gb=4.0
        )
        logger.info(f"Image generated: {output_image_path}")
        return output_image_path

    async def run_audio_diffusion_job(self, audio_texture_prompt: str) -> Path:
        """
        Runs an audio diffusion job (placeholder).
        """
        logger.info(f"Running audio diffusion job for prompt: {audio_texture_prompt}")
        output_audio_path = await self._run_with_resource_check(
            self.audio_diffusion_generator.generate_audio,
            prompt=audio_texture_prompt,
            required_ram_gb=2.0
        )
        logger.info(f"Audio generated: {output_audio_path}")
        return output_audio_path

    # TODO: Implement explicit RAM-aware resource scheduling
    # TODO: Ensure no network calls or telemetry beyond explicit API calls to local services
    # TODO: Integrate with existing UI (requires more info on the UI's communication method)

    def _check_memory_availability(self, required_ram_gb: float) -> bool:
        """
        Placeholder for checking system memory availability. Actual implementation
        would involve querying system resources.
        """
        # This is a placeholder. Real implementation would use platform-specific libraries
        # like `psutil` or `resource` (on Unix-like systems) to get actual memory usage.
        logger.debug(f"Checking for {required_ram_gb} GB RAM availability.")
        # For now, assume sufficient memory is always available.
        return True

    async def _run_with_resource_check(self, func, *args, required_ram_gb: float = 0, **kwargs):
        """
        Wrapper to run a function after checking for resource availability.
        """
        if not self._check_memory_availability(required_ram_gb):
            raise RuntimeError("Insufficient RAM to perform this operation.")
        return await func(*args, **kwargs)
