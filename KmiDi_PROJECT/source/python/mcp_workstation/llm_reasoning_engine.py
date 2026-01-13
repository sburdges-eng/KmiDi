from dataclasses import dataclass, asdict
import json
import textwrap
from typing import Any, Dict, List, Optional

from llama_cpp import Llama

from .audio_generation_engine import AudioGenerationEngine  # Import the new audio engine
from .image_generation_engine import ImageGenerationEngine


@dataclass
class StructuredIntent:
    core_event: Optional[str] = None
    core_resistance: Optional[str] = None
    core_longing: Optional[str] = None
    mood_primary: Optional[str] = None
    vulnerability_scale: Optional[str] = None
    narrative_arc: Optional[str] = None
    technical_genre: Optional[str] = None
    technical_key: Optional[str] = None
    technical_rule_to_break: Optional[str] = None
    midi_plan: Optional[Dict[str, Any]] = None
    image_prompt: Optional[str] = None
    image_style_constraints: Optional[str] = None
    audio_texture_prompt: Optional[str] = None
    explanation: Optional[str] = None
    rule_breaking_logic: Optional[str] = None
    generated_image_data: Optional[Dict[str, Any]] = None
    generated_audio_data: Optional[Dict[str, Any]] = None  # New field for audio results


class LLMReasoningEngine:
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        n_threads: int = 4,
        image_engine: Optional[ImageGenerationEngine] = None,
        audio_engine: Optional[AudioGenerationEngine] = None,
    ):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
        )
        self.image_engine = image_engine or ImageGenerationEngine()
        self.audio_engine = (
            audio_engine or AudioGenerationEngine()
        )  # Initialize AudioGenerationEngine

    def parse_user_intent(self, natural_language_input: str) -> StructuredIntent:
        prompt = textwrap.dedent(
            f"""
            Parse the following natural language input into a structured intent object.
            Provide a JSON output that matches the StructuredIntent dataclass.
            If a field is not inferable, omit it.

            Natural Language Input: {natural_language_input}

            Structured Intent (JSON):
            """
        )
        output = self.llm.create_completion(prompt, max_tokens=500, temperature=0.7, stop=["```"])
        try:
            json_output = output["choices"][0]["text"].strip()
            intent_dict = json.loads(json_output)
            return StructuredIntent(**intent_dict)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing LLM output: {e}")
            return StructuredIntent(explanation=f"Error: Could not parse intent. {e}")

    def expand_midi_prompt(self, structured_intent: StructuredIntent) -> Dict[str, Any]:
        prompt = f"""
        Given the structured intent, expand it into a detailed MIDI plan.
        The plan should be a JSON object suitable for the KmiDi Tier-1 stack.

        Structured Intent: {asdict(structured_intent)}

        Detailed MIDI Plan (JSON):
        """
        output = self.llm.create_completion(prompt, max_tokens=1000, temperature=0.7, stop=["```"])
        try:
            json_output = output["choices"][0]["text"].strip()
            return json.loads(json_output)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error expanding MIDI prompt: {e}")
            return {"error": f"Could not expand MIDI plan. {e}"}

    def generate_image_prompts(self, structured_intent: StructuredIntent) -> StructuredIntent:
        prompt = f"""
        Given the structured intent, generate an image prompt and style constraints for Stable Diffusion 1.5.

        Structured Intent: {asdict(structured_intent)}

        Output format:
        Image Prompt: <prompt>
        Style Constraints: <style_constraints>
        """
        output = self.llm.create_completion(
            prompt, max_tokens=500, temperature=0.7, stop=["Image Prompt:", "Style Constraints:"]
        )

        response_text = output["choices"][0]["text"].strip()
        image_prompt_start = response_text.find("Image Prompt:")
        style_constraints_start = response_text.find("Style Constraints:")

        image_prompt = ""
        style_constraints = ""

        if image_prompt_start != -1:
            if style_constraints_start != -1:
                image_prompt = response_text[
                    image_prompt_start + len("Image Prompt:") : style_constraints_start
                ].strip()
            else:
                image_prompt = response_text[image_prompt_start + len("Image Prompt:") :].strip()

        if style_constraints_start != -1:
            style_constraints = response_text[
                style_constraints_start + len("Style Constraints:") :
            ].strip()

        structured_intent.image_prompt = image_prompt
        structured_intent.image_style_constraints = style_constraints
        return structured_intent

    def generate_image_from_intent(self, structured_intent: StructuredIntent) -> StructuredIntent:
        if structured_intent.image_prompt:
            print(f"Calling image engine with prompt: {structured_intent.image_prompt}")
            image_result = self.image_engine.generate_image(
                prompt=structured_intent.image_prompt,
                style_constraints=structured_intent.image_style_constraints or "",
            )
            structured_intent.generated_image_data = image_result
        else:
            print("No image prompt found in structured intent. Skipping image generation.")
            structured_intent.generated_image_data = {
                "status": "skipped",
                "details": "No image prompt.",
            }
        return structured_intent

    def generate_audio_texture_prompt(
        self, structured_intent: StructuredIntent
    ) -> StructuredIntent:
        prompt = f"""
        Given the structured intent, generate an audio texture prompt for a diffusion model.

        Structured Intent: {asdict(structured_intent)}

        Audio Texture Prompt:
        """
        output = self.llm.create_completion(
            prompt, max_tokens=300, temperature=0.7, stop=["Audio Texture Prompt:"]
        )
        structured_intent.audio_texture_prompt = output["choices"][0]["text"].strip()
        return structured_intent

    def generate_audio_from_intent(self, structured_intent: StructuredIntent) -> StructuredIntent:
        if structured_intent.audio_texture_prompt:
            print(f"Calling audio engine with prompt: {structured_intent.audio_texture_prompt}")
            # Attempt to acquire lock for audio generation. Timeout is optional.
            if self.audio_engine.acquire_lock(timeout=300):
                try:
                    audio_result = self.audio_engine.generate_audio_texture(
                        prompt=structured_intent.audio_texture_prompt
                    )
                    structured_intent.generated_audio_data = audio_result
                finally:
                    self.audio_engine.release_lock()
            else:
                print("Could not acquire lock for audio generation. Skipping.")
                structured_intent.generated_audio_data = {
                    "status": "skipped",
                    "details": "Could not acquire audio generation lock.",
                }
        else:
            print("No audio texture prompt found in structured intent. Skipping audio generation.")
            structured_intent.generated_audio_data = {
                "status": "skipped",
                "details": "No audio texture prompt.",
            }
        return structured_intent

    def generate_explanation(
        self, original_intent: str, generated_output: Any, output_type: str
    ) -> str:
        prompt = textwrap.dedent(
            f"""
            Explain the reasoning behind the generated {output_type} based on the original user intent.

            Original Intent: {original_intent}
            Generated {output_type}: {json.dumps(generated_output, indent=2)}

            Explanation:
            """
        )
        output = self.llm.create_completion(
            prompt, max_tokens=500, temperature=0.7, stop=["Explanation:"]
        )
        return output["choices"][0]["text"].strip()

    def get_rule_breaking_logic(self, structured_intent: StructuredIntent) -> str:
        if structured_intent.technical_rule_to_break:
            return f"Apply rule-breaking logic: {structured_intent.technical_rule_to_break}"
        return "No explicit rule-breaking requested."
