"""
BrainController
---------------
Local reasoning controller using Mistral 7B (GGUF) via llama.cpp (Metal).

Responsibilities (thinking only):
 - Parse raw user intent text into CompleteSongIntent
 - Expand prompts for downstream generators (MIDI, image, audio texture)
 - Explain decisions and rule-breaking explicitly
 - Produce structured, inspectable outputs (no side effects, no generation)

Strictly forbidden:
 - Audio/MIDI/image generation
 - Network calls or telemetry
 - Background services
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any, List
import json
import os
import time

from music_brain.session.intent_schema import (
    CompleteSongIntent,
    suggest_rule_break,
    get_rule_breaking_info,
)


def _try_import_llama() -> Optional[Any]:
    """
    Lazy import to keep the module importable even if llama-cpp-python
    is not installed. Returns the Llama class or None.
    """
    try:
        from llama_cpp import Llama

        return Llama
    except Exception:
        return None


@dataclass
class BrainConfig:
    """Configuration for local Mistral 7B reasoning."""

    model_path: Path
    n_ctx: int = 4096
    n_threads: int = max(1, os.cpu_count() or 1)
    n_gpu_layers: int = 35  # tuned for 7B on Metal; adjust if needed
    temperature: float = 0.2  # low for structured outputs
    top_p: float = 0.9
    seed: Optional[int] = None
    metal: bool = True  # enable Metal backend on Apple Silicon


class BrainController:
    """
    Local-first reasoning controller wrapping llama.cpp (Mistral 7B GGUF).
    """

    def __init__(self, config: BrainConfig):
        self.config = config
        self._llama_cls = _try_import_llama()
        self._llm = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def load(self):
        """Load the Mistral model into memory (Metal if available)."""
        if self._llm is not None:
            return
        if not self._llama_cls:
            raise ImportError(
                "llama-cpp-python is required for BrainController. "
                "Install with `pip install llama-cpp-python`."
            )
        if not self.config.model_path.exists():
            raise FileNotFoundError(f"GGUF model not found at {self.config.model_path}")

        self._llm = self._llama_cls(
            model_path=str(self.config.model_path),
            n_ctx=self.config.n_ctx,
            n_threads=self.config.n_threads,
            n_gpu_layers=self.config.n_gpu_layers,
            seed=self.config.seed,
            logits_all=False,
            verbose=False,
            use_mmap=True,
            use_mlock=False,
            # Metal acceleration is enabled automatically on Apple Silicon if built with metal
        )

    def unload(self):
        """Release model resources."""
        self._llm = None

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #
    def parse_intent(self, user_text: str) -> CompleteSongIntent:
        """
        Parse free-form user input into a structured CompleteSongIntent.
        """
        prompt = self._build_intent_prompt(user_text)
        response = self._run_chat(prompt, temperature=self.config.temperature)
        data = self._extract_json(response) or {}
        intent = CompleteSongIntent.from_dict(data) if data else CompleteSongIntent()
        return intent

    def expand_prompts(self, intent: CompleteSongIntent) -> Dict[str, str]:
        """
        Expand prompts for downstream generators (image, audio texture, MIDI hints).
        """
        prompt = self._build_expansion_prompt(intent)
        response = self._run_chat(prompt, temperature=0.4)
        data = self._extract_json(response) or {}
        return {
            "image_prompt": data.get("image_prompt", ""),
            "image_style_rules": json.dumps(data.get("image_style_rules", {})),
            "audio_texture_prompt": data.get("audio_texture_prompt", ""),
            "midi_hints": json.dumps(data.get("midi_hints", {})),
        }

    def explain_decision(self, intent: CompleteSongIntent, decision: str) -> str:
        """
        Provide an explicit explanation for a decision (e.g., chord choice).
        """
        prompt = self._build_explain_prompt(intent, decision)
        return self._run_chat(prompt, temperature=0.5)

    def suggest_rule_breaks(self, intent: CompleteSongIntent) -> List[Dict[str, str]]:
        """
        Suggest explicit rule-breaking options grounded in the target emotion.
        """
        emotion = intent.song_intent.mood_primary or ""
        suggestions = suggest_rule_break(emotion)
        # Attach reasoning using the LLM (but do not mutate rules implicitly)
        if suggestions:
            explain_prompt = self._build_rule_break_prompt(emotion, suggestions)
            explanation = self._run_chat(explain_prompt, temperature=0.5)
            return [
                {**s, "llm_explanation": explanation}
                for s in suggestions
            ]
        return suggestions

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _run_chat(self, prompt: str, temperature: float) -> str:
        if not self._llm:
            raise RuntimeError("BrainController not loaded. Call load() before use.")

        start = time.time()
        result = self._llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=self.config.top_p,
            max_tokens=None,
        )
        elapsed = (time.time() - start) * 1000
        choice = (result.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content", "") or ""
        # Debug log could be added here; keeping output minimal/explicit
        return content.strip()

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        """
        Extract the first JSON object from a text blob. If none found, return None.
        """
        if not text:
            return None
        try:
            # simple fast-path
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # fallback: find fenced JSON
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                cleaned = part.strip()
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    continue
        return None

    # Prompt builders (kept minimal and explicit)
    def _build_intent_prompt(self, user_text: str) -> str:
        return (
            "Parse the user's request into structured song intent. "
            "Return ONLY JSON matching CompleteSongIntent fields:\n"
            "{\n"
            '  "title": "",\n'
            '  "song_root": {\n'
            '    "core_event": "",\n'
            '    "core_resistance": "",\n'
            '    "core_longing": "",\n'
            '    "core_stakes": "",\n'
            '    "core_transformation": ""\n'
            "  },\n"
            '  "song_intent": {\n'
            '    "mood_primary": "",\n'
            '    "mood_secondary_tension": 0.5,\n'
            '    "imagery_texture": "",\n'
            '    "vulnerability_scale": "Medium",\n'
            '    "narrative_arc": ""\n'
            "  },\n"
            '  "technical_constraints": {\n'
            '    "technical_genre": "",\n'
            '    "technical_tempo_range": [80,120],\n'
            '    "technical_key": "",\n'
            '    "technical_mode": "",\n'
            '    "technical_groove_feel": "",\n'
            '    "technical_rule_to_break": "",\n'
            '    "rule_breaking_justification": ""\n'
            "  },\n"
            '  "system_directive": {\n'
            '    "output_target": "midi",\n'
            '    "output_feedback_loop": ""\n'
            "  }\n"
            "}\n"
            "User request:\n"
            f"{user_text}"
        )

    def _build_expansion_prompt(self, intent: CompleteSongIntent) -> str:
        return (
            "Given the structured song intent, produce generation-ready prompts. "
            "Return ONLY JSON with keys: image_prompt (string), "
            "image_style_rules (object), audio_texture_prompt (string), midi_hints (object). "
            "Do not generate media; only describe instructions.\n"
            f"Intent JSON:\n{json.dumps(intent.to_dict(), indent=2)}"
        )

    def _build_explain_prompt(
        self, intent: CompleteSongIntent, decision: str
    ) -> str:
        return (
            "Explain concisely why this decision is aligned with the song intent. "
            "Focus on harmony, groove, texture, or narrative fit. "
            "Limit to 3 sentences.\n"
            f"Decision: {decision}\n"
            f"Intent: {json.dumps(intent.to_dict(), indent=2)}"
        )

    def _build_rule_break_prompt(
        self, emotion: str, suggestions: List[Dict[str, str]]
    ) -> str:
        return (
            "Provide a short justification (1-2 sentences) for each rule-breaking option, "
            "grounded in the target emotion. Do not add new rules. "
            f"Emotion: {emotion}\n"
            f"Rules: {json.dumps(suggestions, indent=2)}"
        )


# Convenience factory
def create_brain_controller(
    model_path: str,
    n_ctx: int = 4096,
    seed: Optional[int] = None,
) -> BrainController:
    config = BrainConfig(
        model_path=Path(model_path),
        n_ctx=n_ctx,
        seed=seed,
    )
    return BrainController(config)

