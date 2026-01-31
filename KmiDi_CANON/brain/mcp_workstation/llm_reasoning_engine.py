"""
LLM reasoning engine — parses user text into nested CompleteSongIntent.

When KMI_DI_LLM_MODEL_PATH / LLM_MODEL_PATH points to a GGUF model and
llama-cpp-python is available, uses the model for intent parsing.
Otherwise uses a rule-based keyword parser (fallback) so the orchestrator
always gets a valid nested intent. Image/audio prompts are derived from
intent fields (template-based) or from LLM when available.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from music_brain.session.intent_schema import (
    CompleteSongIntent,
    SongRoot,
    SongIntent,
    TechnicalConstraints,
    SystemDirective,
)

logger = logging.getLogger(__name__)

# Env: set to GGUF model path for LLM-backed parsing; fallback to rule-based if unset or load fails.
MODEL_PATH_ENV_KEYS = ("KMI_DI_LLM_MODEL_PATH", "LLM_MODEL_PATH")

# Rule-based: keyword → mood_primary (Phase 1). Order matters: more specific first (e.g. nostalgic before want).
MOOD_KEYWORDS = [
    ("nostalgia", ["nostalgia", "nostalgic", "memory", "remember", "past", "childhood"]),
    ("grief", ["grief", "loss", "mourning", "sad", "miss", "gone"]),
    ("defiance", ["defiance", "angry", "rage", "fight", "refuse", "won't"]),
    ("tenderness", ["tenderness", "gentle", "soft", "love", "warm", "kind"]),
    ("melancholy", ["melancholy", "blue", "sombre", "dark", "heavy"]),
    ("hope", ["hope", "hopeful", "bright", "lift", "rise"]),
    ("longing", ["longing", "yearning", "wish"]),  # "want" omitted — too generic
]

# Rule-based: phrase → technical_key (Phase 2)
KEY_PATTERNS = [
    (r"\b(in\s+)?([A-G])\s*(major|maj)\b", lambda m: (m.group(2), "major")),
    (r"\b([A-G])b\s*(major|maj)\b", lambda m: (m.group(1) + "b", "major")),
    (r"\b([A-G])#\s*(major|maj)\b", lambda m: (m.group(1) + "#", "major")),
    (r"\b(in\s+)?([A-G])\s*minor\b", lambda m: (m.group(2), "minor")),
    (r"\b([A-G])m\b", lambda m: (m.group(1), "minor")),
    (r"\b([A-G])b\s*minor\b", lambda m: (m.group(1) + "b", "minor")),
    (r"\b([A-G])#\s*minor\b", lambda m: (m.group(1) + "#", "minor")),
    (r"\b(C|D|E|F|G|A|B)\b", lambda m: (m.group(1), "major")),
]

# Rule-based: keyword → technical_rule_to_break
RULE_BREAK_KEYWORDS = {
    "HARMONY_ModalInterchange": ["borrow", "borrowed", "modal", "color", "bittersweet", "nostalgic"],
    "HARMONY_AvoidTonicResolution": ["unresolved", "yearning", "open", "don't resolve"],
    "HARMONY_UnresolvedDissonance": ["dissonant", "tension", "unresolved", "ache"],
    "RHYTHM_TempoFluctuation": ["breathing", "human", "organic", "rubato"],
    "RHYTHM_ConstantDisplacement": ["off-kilter", "displaced", "uneasy"],
}


def _rule_based_parse(user_intent_text: str) -> CompleteSongIntent:
    """Build nested CompleteSongIntent from keyword/pattern rules. No LLM."""
    text = (user_intent_text or "").strip().lower()
    if not text:
        return CompleteSongIntent(
            explanation="Empty intent; using defaults (F major).",
        )

    # SongRoot: use first sentence or first 200 chars as core_event; rest as context
    parts = text.split(".", 1)
    core_event = (parts[0].strip() or text[:200])[:500]
    core_longing = ""
    for mood, keywords in MOOD_KEYWORDS:
        if any(k in text for k in keywords):
            core_longing = mood
            break

    # SongIntent: mood, tension, narrative
    mood_primary = "neutral"
    for mood, keywords in MOOD_KEYWORDS:
        if any(k in text for k in keywords):
            mood_primary = mood
            break
    narrative_arc = "Build and Release"
    if "slow" in text or "gradual" in text:
        narrative_arc = "Slow Reveal"
    if "repetitive" in text or "loop" in text:
        narrative_arc = "Repetitive Despair"

    # TechnicalConstraints: key, mode, rule to break
    technical_key, technical_mode = "F", "major"
    for pattern, getter in KEY_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                k, mode = getter(m)
                if isinstance(k, str) and k:
                    k = k.strip()
                    technical_key = k[0].upper() + (k[1:] if len(k) > 1 else "")
                if isinstance(mode, str) and mode:
                    technical_mode = mode.strip().lower()
                break
            except (IndexError, AttributeError):
                pass
    technical_rule_to_break = ""
    rule_breaking_justification = ""
    for rule, keywords in RULE_BREAK_KEYWORDS.items():
        if any(k in text for k in keywords):
            technical_rule_to_break = rule
            rule_breaking_justification = f"User text suggests {rule}; applied from keywords."
            break

    intent = CompleteSongIntent(
        song_root=SongRoot(
            core_event=core_event,
            core_resistance="",
            core_longing=core_longing or mood_primary,
            core_stakes="",
            core_transformation="",
        ),
        song_intent=SongIntent(
            mood_primary=mood_primary,
            mood_secondary_tension=0.5,
            imagery_texture="",
            vulnerability_scale="Medium",
            narrative_arc=narrative_arc,
        ),
        technical_constraints=TechnicalConstraints(
            technical_genre="",
            technical_tempo_range=(80, 120),
            technical_key=technical_key,
            technical_mode=technical_mode,
            technical_groove_feel="",
            technical_rule_to_break=technical_rule_to_break,
            rule_breaking_justification=rule_breaking_justification,
        ),
        system_directive=SystemDirective(),
        explanation=f"Rule-based parse: mood={mood_primary}, key={technical_key} {technical_mode}.",
    )
    return intent


def _llm_parse_if_available(user_intent_text: str, model_path: Path) -> Optional[CompleteSongIntent]:
    """If llama-cpp-python and model exist, prompt for JSON intent; else None."""
    if not model_path.exists() or not model_path.is_file():
        return None
    try:
        from llama_cpp import Llama
    except ImportError:
        logger.debug("llama-cpp-python not installed; using rule-based parse.")
        return None

    prompt = f"""Extract song intent from this user request. Reply with ONLY a JSON object (no markdown, no explanation) with these keys:
song_root: {{ "core_event": str, "core_resistance": str, "core_longing": str, "core_stakes": str, "core_transformation": str }}
song_intent: {{ "mood_primary": str, "mood_secondary_tension": float 0-1, "imagery_texture": str, "vulnerability_scale": "Low"|"Medium"|"High", "narrative_arc": str }}
technical_constraints: {{ "technical_genre": str, "technical_tempo_range": [int,int], "technical_key": str e.g. "F", "technical_mode": "major"|"minor", "technical_groove_feel": str, "technical_rule_to_break": str e.g. "HARMONY_ModalInterchange", "rule_breaking_justification": str }}

User request: {user_intent_text[:800]}
JSON:"""

    try:
        llm = Llama(model_path=str(model_path), verbose=False)
        out = llm(prompt, max_tokens=512, temperature=0.2, stop=["}\n"])
        raw = (out.get("choices") or [{}])[0].get("text", "")
        # Try to find JSON object in response
        start = raw.find("{")
        if start == -1:
            return None
        end = raw.rfind("}") + 1
        if end <= start:
            return None
        data = json.loads(raw[start:end])
        # Build nested structure for from_dict
        if "song_root" not in data:
            data["song_root"] = {}
        if "song_intent" not in data:
            data["song_intent"] = {}
        if "technical_constraints" not in data:
            data["technical_constraints"] = {}
        return CompleteSongIntent.from_dict(data)
    except Exception as e:
        logger.warning("LLM parse failed: %s; falling back to rule-based.", e)
        return None


class LLMReasoningEngine:
    """
    Parses user text into nested CompleteSongIntent.
    Uses LLM when model path is set and loadable; otherwise rule-based fallback.
    """

    def __init__(
        self,
        model_path: str,
        image_engine: Any = None,
        audio_engine: Any = None,
    ):
        self.model_path = Path(model_path) if model_path else Path()
        self.image_engine = image_engine
        self.audio_engine = audio_engine
        self._llm = None  # Lazy load when first needed

    def parse_user_intent(self, user_intent_text: str) -> CompleteSongIntent:
        """Parse user text into nested CompleteSongIntent. LLM if available else rule-based."""
        intent = _llm_parse_if_available(user_intent_text, self.model_path)
        if intent is not None:
            intent.explanation = "LLM parse."
            return intent
        return _rule_based_parse(user_intent_text)

    def generate_image_prompts(self, structured_intent: CompleteSongIntent) -> CompleteSongIntent:
        """Derive image_prompt and image_style_constraints from intent (template-based)."""
        mood = structured_intent.song_intent.mood_primary or "neutral"
        imagery = structured_intent.song_intent.imagery_texture or ""
        narrative = structured_intent.song_intent.narrative_arc or ""
        structured_intent.image_prompt = (
            f"Album art: {mood}, {imagery}. Mood: {narrative}."
        ).strip(" ,.")
        structured_intent.image_style_constraints = "Emotional, music-focused, no text."
        return structured_intent

    def generate_audio_texture_prompt(
        self, structured_intent: CompleteSongIntent
    ) -> CompleteSongIntent:
        """Derive audio_texture_prompt from intent (template-based)."""
        mood = structured_intent.song_intent.mood_primary or "neutral"
        imagery = structured_intent.song_intent.imagery_texture or ""
        structured_intent.audio_texture_prompt = f"Texture: {mood}, {imagery}."
        return structured_intent

    def generate_image_from_intent(
        self, structured_intent: CompleteSongIntent
    ) -> CompleteSongIntent:
        """Call image_engine.generate with intent prompts; set generated_image_data (stubbed/skipped if no engine or no prompt)."""
        if not self.image_engine:
            structured_intent.generated_image_data = {"status": "stubbed", "details": "No image engine configured."}
            return structured_intent
        prompt = getattr(structured_intent, "image_prompt", "") or ""
        if not prompt:
            structured_intent.generated_image_data = {"status": "skipped", "details": "No image_prompt in intent."}
            return structured_intent
        style = getattr(structured_intent, "image_style_constraints", "") or ""
        try:
            result = self.image_engine.generate(
                prompt=prompt,
                style_constraints=style,
                assume_locked=False,
                lock_timeout=300.0,
            )
            structured_intent.generated_image_data = result or {}
        except Exception as e:
            logger.warning("Image engine generate failed: %s", e)
            structured_intent.generated_image_data = {"status": "error", "details": str(e)}
        return structured_intent

    def generate_audio_from_intent(
        self, structured_intent: CompleteSongIntent
    ) -> CompleteSongIntent:
        """Call audio_engine.generate_audio_texture with intent prompt; set generated_audio_data (stubbed/skipped if no engine or no prompt)."""
        if not self.audio_engine:
            structured_intent.generated_audio_data = {"status": "stubbed", "details": "No audio engine configured."}
            return structured_intent
        prompt = getattr(structured_intent, "audio_texture_prompt", "") or ""
        if not prompt:
            structured_intent.generated_audio_data = {"status": "skipped", "details": "No audio_texture_prompt in intent."}
            return structured_intent
        try:
            result = self.audio_engine.generate_audio_texture(
                prompt=prompt,
                assume_locked=False,
                lock_timeout=300.0,
            )
            structured_intent.generated_audio_data = result or {}
        except Exception as e:
            logger.warning("Audio engine generate_audio_texture failed: %s", e)
            structured_intent.generated_audio_data = {"status": "error", "details": str(e)}
        return structured_intent
