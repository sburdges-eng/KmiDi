import argparse
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .audio_generation_engine import AudioGenerationEngine
from .image_generation_engine import ImageGenerationEngine
from .llm_reasoning_engine import LLMReasoningEngine, StructuredIntent
from music_brain.session.intent_schema import CompleteSongIntent
from music_brain.tier1.midi_pipeline_wrapper import MIDIGenerationPipeline

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")


class Orchestrator:
    def __init__(self, llm_model_path: str, output_dir: str = "./orchestrator_outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.image_engine = ImageGenerationEngine(
            model_dir=str(self.output_dir / "stable_diffusion_v1_5")
        )
        self.audio_engine = AudioGenerationEngine(
            output_dir=str(self.output_dir / "audio_textures")
        )

        self.llm_engine = LLMReasoningEngine(
            model_path=llm_model_path,
            image_engine=self.image_engine,  # Pass image engine to LLM
            audio_engine=self.audio_engine,  # Pass audio engine to LLM
        )
        self.midi_pipeline = MIDIGenerationPipeline()

        self.resource_locks = {
            "llm": threading.Lock(),
            "midi_gen": threading.Lock(),
            "image_gen": threading.Lock(),
            "audio_gen": threading.Lock(),
        }

        logging.info("Orchestrator initialized.")

    def _acquire_resource(self, resource_name: str, timeout: float = 300) -> bool:
        logging.info(f"Attempting to acquire lock for {resource_name}...")
        if self.resource_locks[resource_name].acquire(timeout=timeout):
            logging.info(f"Lock acquired for {resource_name}.")
            return True
        logging.warning(f"Failed to acquire lock for {resource_name} within {timeout} seconds.")
        return False

    def _release_resource(self, resource_name: str):
        try:
            self.resource_locks[resource_name].release()
            logging.info(f"Lock released for {resource_name}.")
        except RuntimeError:
            logging.warning(
                "Tried to release lock for %s but it was not held by this thread.",
                resource_name,
            )

    def execute_workflow(
        self, user_intent_text: str, enable_image_gen: bool = True, enable_audio_gen: bool = False
    ) -> CompleteSongIntent:
        logging.info(f"Starting workflow for intent: '{user_intent_text}'")
        start_time = time.time()

        # Phase 1: LLM Reasoning (Intent Parsing, Prompt Expansion)
        if not self._acquire_resource("llm"):
            raise RuntimeError("Could not acquire LLM resource.")
        try:
            structured_intent = self.llm_engine.parse_user_intent(user_intent_text)
            structured_intent = self.llm_engine.generate_image_prompts(
                structured_intent
            )  # Generates image prompts based on intent
            if enable_audio_gen:
                structured_intent = self.llm_engine.generate_audio_texture_prompt(
                    structured_intent
                )  # Generates audio prompts
            logging.info("LLM reasoning complete.")
        finally:
            self._release_resource("llm")

        # Convert StructuredIntent to CompleteSongIntent for MIDI pipeline compatibility
        # This is a simplification; a more robust mapping would be needed.
        complete_intent = CompleteSongIntent(
            core_event=structured_intent.core_event,
            mood_primary=structured_intent.mood_primary,
            technical_genre=structured_intent.technical_genre,
            technical_key=structured_intent.technical_key,
            technical_rule_to_break=structured_intent.technical_rule_to_break,
            midi_plan=structured_intent.midi_plan,
            image_prompt=structured_intent.image_prompt,
            image_style_constraints=structured_intent.image_style_constraints,
            audio_texture_prompt=structured_intent.audio_texture_prompt,
            explanation=structured_intent.explanation,
            rule_breaking_logic=structured_intent.rule_breaking_logic,
            # Additional fields need to be mapped if used by MIDI pipeline
        )

        # Phase 2: MIDI Generation
        # MIDI generation is fast and deterministic, so it might not need a separate lock if it doesn't contend with LLM.
        # However, for safety and consistency in a multi-resource environment, it's good to include it.
        if not self._acquire_resource("midi_gen"):
            raise RuntimeError("Could not acquire MIDI resource.")
        try:
            midi_result = self.midi_pipeline.generate_midi(
                complete_intent, output_dir=str(self.output_dir / "midi_outputs")
            )
            complete_intent.midi_plan = midi_result
            logging.info("MIDI generation complete.")
        finally:
            self._release_resource("midi_gen")

        # Phase 3: Image Generation (Optional)
        if enable_image_gen and structured_intent.image_prompt:
            # The LLMEngine already holds a reference to image_engine, so its internal lock is used.
            # Orchestrator handles top-level resource scheduling.
            if not self._acquire_resource("image_gen"):
                raise RuntimeError("Could not acquire Image Generation resource.")
            try:
                structured_intent = self.llm_engine.generate_image_from_intent(structured_intent)
                complete_intent.generated_image_data = structured_intent.generated_image_data
                logging.info("Image generation complete.")
            finally:
                self._release_resource("image_gen")

        # Phase 4: Audio Generation (Optional)
        if enable_audio_gen and structured_intent.audio_texture_prompt:
            # The LLMEngine already holds a reference to audio_engine, so its internal lock is used.
            # Orchestrator handles top-level resource scheduling.
            # This requires mutual exclusion, which is handled by the AudioGenerationEngine's internal lock.
            # The orchestrator's resource_lock["audio_gen"] ensures only one audio job runs at a time.
            if not self._acquire_resource("audio_gen"):
                raise RuntimeError("Could not acquire Audio Generation resource.")
            try:
                structured_intent = self.llm_engine.generate_audio_from_intent(structured_intent)
                complete_intent.generated_audio_data = structured_intent.generated_audio_data
                logging.info("Audio generation complete.")
            finally:
                self._release_resource("audio_gen")

        end_time = time.time()
        logging.info(f"Workflow completed in {end_time - start_time:.2f} seconds.")
        return complete_intent


def main():
    parser = argparse.ArgumentParser(description="KmiDi Local Metal AI Orchestrator CLI")
    parser.add_argument(
        "--llm_model_path", type=str, required=True, help="Path to the Mistral 7B GGUF model file."
    )
    parser.add_argument("--prompt", type=str, required=True, help="Natural language user intent.")
    parser.add_argument("--no_image_gen", action="store_true", help="Disable image generation.")
    parser.add_argument(
        "--enable_audio_gen", action="store_true", help="Enable optional audio texture generation."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./orchestrator_outputs",
        help="Directory for all generated outputs.",
    )

    args = parser.parse_args()

    orchestrator = Orchestrator(llm_model_path=args.llm_model_path, output_dir=args.output_dir)

    try:
        final_intent = orchestrator.execute_workflow(
            user_intent_text=args.prompt,
            enable_image_gen=not args.no_image_gen,
            enable_audio_gen=args.enable_audio_gen,
        )
        print("\n--- Workflow Results ---")
        print(f"User Intent: {args.prompt}")
        midi_status = (
            final_intent.midi_plan.get("status", "N/A")
            if isinstance(final_intent.midi_plan, dict)
            else "N/A"
        )
        image_status = (
            final_intent.generated_image_data.get("status", "disabled")
            if isinstance(final_intent.generated_image_data, dict)
            else "disabled"
        )
        audio_status = (
            final_intent.generated_audio_data.get("status", "disabled")
            if isinstance(final_intent.generated_audio_data, dict)
            else "disabled"
        )
        print(f"Generated MIDI Plan Status: {midi_status}")
        print(f"Generated Image Status: {image_status}")
        print(f"Generated Audio Status: {audio_status}")
        print(f"Explanation: {final_intent.explanation}")

        # Optionally save the full intent object to JSON for inspection
        intent_output_path = Path(args.output_dir) / "final_intent.json"
        final_intent.save(str(intent_output_path))
        print(f"Full intent saved to: {intent_output_path}")

    except RuntimeError as e:
        logging.error(f"Workflow failed: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()
