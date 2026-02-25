from dataclasses import dataclass, asdict
import json
import textwrap
from typing import Any, Dict, Optional

try:
    from llama_cpp import Llama
except ImportError:  # pragma: no cover - optional dependency for tests
    Llama = None

    class _MissingLlama:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "llama_cpp is required for LLMReasoningEngine. "
                "Install via `pip install llama-cpp-python` (Metal enabled on macOS)."
            )

from .audio_generation_engine import AudioGenerationEngine  # Import the new audio engine
from .image_generation_engine import ImageGenerationEngine


@dataclass
class StructuredIntent:
    core_event: Optional[str] = None
    core_resistance: Optional[str] = None
    core_longing: Optional[str] = None
    core_stakes: Optional[str] = None
    core_transformation: Optional[str] = None
    mood_primary: Optional[str] = None
    mood_secondary_tension: Optional[str] = None
    imagery_texture: Optional[str] = None
    vulnerability_scale: Optional[str] = None
    narrative_arc: Optional[str] = None
    technical_genre: Optional[str] = None
    technical_key: Optional[str] = None
    technical_mode: Optional[str] = None
    technical_groove_feel: Optional[str] = None
    technical_rule_to_break: Optional[str] = None
    rule_breaking_justification: Optional[str] = None
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
        if Llama is None:
            raise ImportError(
                "llama_cpp is not installed; install llama-cpp-python to use LLMReasoningEngine."
            )
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
        try:
            output = self.llm.create_completion(
                prompt, max_tokens=500, temperature=0.7, stop=["```"]
            )
            if not output or "choices" not in output:
                raise ValueError("LLM returned empty or invalid response")
            if not output["choices"] or "text" not in output["choices"][0]:
                raise ValueError("LLM response missing text field")
            json_output = output["choices"][0]["text"].strip()
            # Try to extract JSON if wrapped in code blocks
            if "```json" in json_output:
                json_output = json_output.split("```json")[1].split("```")[0].strip()
            elif "```" in json_output:
                json_output = json_output.split("```")[1].split("```")[0].strip()
            intent_dict = json.loads(json_output)
            return StructuredIntent(**intent_dict)
        except (json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
            print(f"Error parsing LLM output: {e}")
            return StructuredIntent(explanation=f"Error: Could not parse intent. {e}")

    def expand_midi_prompt(self, structured_intent: StructuredIntent) -> Dict[str, Any]:
        prompt = f"""
        Given the structured intent, expand it into a detailed MIDI plan.
        The plan should be a JSON object suitable for the KmiDi Tier-1 stack.

        Structured Intent: {asdict(structured_intent)}

        Detailed MIDI Plan (JSON):
        """
        try:
            output = self.llm.create_completion(
                prompt, max_tokens=1000, temperature=0.7, stop=["```"]
            )
            if not output or "choices" not in output:
                raise ValueError("LLM returned empty or invalid response")
            if not output["choices"] or "text" not in output["choices"][0]:
                raise ValueError("LLM response missing text field")
            json_output = output["choices"][0]["text"].strip()
            # Try to extract JSON if wrapped in code blocks
            if "```json" in json_output:
                json_output = json_output.split("```json")[1].split("```")[0].strip()
            elif "```" in json_output:
                json_output = json_output.split("```")[1].split("```")[0].strip()
            return json.loads(json_output)
        except (json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
            print(f"Error expanding MIDI prompt: {e}")
            return {"error": f"Could not expand MIDI plan. {e}"}

    def generate_image_prompts(self, structured_intent: StructuredIntent) -> StructuredIntent:
        # Preserve any prompts already inferred during parsing to avoid wiping them on fallback.
        current_prompt = structured_intent.image_prompt or ""
        current_style = structured_intent.image_style_constraints or ""

        prompt = textwrap.dedent(
            f"""
            Given the structured intent, return a JSON object with two fields:
            - image_prompt: a concise prompt for Stable Diffusion 1.5
            - style_constraints: short style notes (may be empty)

            Structured Intent: {asdict(structured_intent)}

            JSON:
            """
        )

        try:
            output = self.llm.create_completion(
                prompt, max_tokens=400, temperature=0.7
            )
            if not output or "choices" not in output:
                raise ValueError("LLM returned empty or invalid response")
            if not output["choices"] or "text" not in output["choices"][0]:
                raise ValueError("LLM response missing text field")
            response_text = output["choices"][0]["text"].strip()
        except (KeyError, IndexError, TypeError, ValueError) as e:
            print(f"Error reading image prompt choices: {e}")
            structured_intent.image_prompt = current_prompt
            structured_intent.image_style_constraints = current_style
            return structured_intent

        try:
            parsed = json.loads(response_text)
            structured_intent.image_prompt = (parsed.get("image_prompt") or current_prompt).strip()
            structured_intent.image_style_constraints = (
                parsed.get("style_constraints") or current_style
            ).strip()
        except (json.JSONDecodeError, AttributeError):
            # Attempt a simple fallback parse from plain text responses
            lines = [line.strip() for line in response_text.splitlines() if line.strip()]
            for line in lines:
                if line.lower().startswith("image prompt:"):
                    current_prompt = line.split(":", 1)[1].strip() or current_prompt
                if line.lower().startswith("style constraints:"):
                    current_style = line.split(":", 1)[1].strip() or current_style
            structured_intent.image_prompt = current_prompt
            structured_intent.image_style_constraints = current_style

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
        current_prompt = structured_intent.audio_texture_prompt or ""
        prompt = textwrap.dedent(
            f"""
            Given the structured intent, return a JSON object with one field:
            - audio_texture_prompt: a concise prompt for an audio diffusion model

            Structured Intent: {asdict(structured_intent)}

            JSON:
            """
        )
        try:
            output = self.llm.create_completion(
                prompt, max_tokens=300, temperature=0.7
            )
            if not output or "choices" not in output:
                raise ValueError("LLM returned empty or invalid response")
            if not output["choices"] or "text" not in output["choices"][0]:
                raise ValueError("LLM response missing text field")
            response_text = output["choices"][0]["text"].strip()
        except (KeyError, IndexError, TypeError, ValueError) as e:
            print(f"Error reading audio prompt choices: {e}")
            structured_intent.audio_texture_prompt = current_prompt
            return structured_intent
        try:
            parsed = json.loads(response_text)
            structured_intent.audio_texture_prompt = (
                parsed.get("audio_texture_prompt") or current_prompt
            ).strip()
        except (json.JSONDecodeError, AttributeError):
            # Fallback: handle simple "Audio Texture Prompt: ..." responses
            lines = [line.strip() for line in response_text.splitlines() if line.strip()]
            for line in lines:
                if line.lower().startswith("audio texture prompt:"):
                    current_prompt = line.split(":", 1)[1].strip() or current_prompt
            structured_intent.audio_texture_prompt = current_prompt
        return structured_intent

    def generate_audio_from_intent(self, structured_intent: StructuredIntent) -> StructuredIntent:
        if structured_intent.audio_texture_prompt:
            print(f"Calling audio engine with prompt: {structured_intent.audio_texture_prompt}")
            # Use timeout-based lock acquisition to prevent indefinite blocking.
            audio_result = self.audio_engine.generate_audio_texture(
                prompt=structured_intent.audio_texture_prompt,
                assume_locked=False,
                lock_timeout=300.0,  # 5 minute timeout to prevent workflow hangs
            )
            structured_intent.generated_audio_data = audio_result
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
