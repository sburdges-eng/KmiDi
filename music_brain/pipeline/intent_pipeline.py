"""
Deterministic 3-stage intent pipeline.

Stage 1: Normalize — raw/unstructured input → intermediate dict (emotion_map, fallbacks).
Stage 2: Validate — intermediate dict → CompleteSongIntentRequest (Pydantic).
Stage 3: Expand — request + validated → CompleteSongIntent (full internal fields; imagery_texture only here).

Precedence: validated > normalized > raw request.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from music_brain.utils.serialization import pydantic_to_dict
from music_brain.theory.constants import VALID_MUSICAL_MODES

from music_brain.engine_api.schema import CompleteSongIntentRequest
from music_brain.session.intent_schema import CompleteSongIntent

# -----------------------------------------------------------------------------
# Stage 1 constants (do not override explicitly provided request values)
# -----------------------------------------------------------------------------

# Canonical emotion mapping for inferring mood_primary when not explicitly provided.
# Used only in Stage 1 (Normalize).
EMOTION_MAP = {
    "grief": "grief",
    "sadness": "grief",
    "joy": "tenderness",
    "happiness": "tenderness",
    "anger": "rage",
    "rage": "rage",
    "fear": "fear",
    "love": "tenderness",
    "nostalgia": "nostalgia",
    "awe": "awe",
}

# Defaults for required schema fields when missing from request (infer only).
DEFAULT_KEY_MODE = "C major"
DEFAULT_TEMPO = 82
DEFAULT_STRUCTURE = [{"name": "verse", "bars": 8, "repetitions": 1}]
DEFAULT_INSTRUMENTS = [{"instrument": "piano", "techniques": []}]
DEFAULT_NARRATIVE_ARC = "Climb-to-Climax"
DEFAULT_GROOVE_FEEL = "Straight/Driving"

# Keys passed to CompleteSongIntentRequest (extras like energy_curve are stripped).
_ALLOWED_COMPLETE = frozenset(
    {
        "core_desire",
        "mood_primary",
        "genre",
        "tempo",
        "key_mode",
        "structure",
        "instruments",
        "allow_legacy_fallback",
        "groove_feel",
        "narrative_arc",
        "rule_to_break",
        "rule_justification",
    }
)


def _tech_dict(request: Any) -> Dict[str, Any]:
    """Get technical payload as dict. Does not override; only provides access."""
    tech = getattr(request.intent, "technical", None) or {}
    return pydantic_to_dict(tech)


def _get_str(obj: Any, key: str, default: str = "") -> str:
    """Safe string from dict or object. Returns default if missing or empty."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        val = obj.get(key, default)
    else:
        val = getattr(obj, key, default)
    return (str(val).strip() if val is not None else default) or default


def _infer_mood_primary(emotional: str) -> str:
    """
    Infer canonical mood_primary from emotional_intent string.
    Stage 1 only: emotion_map + strip parenthetical.
    """
    emotional = (emotional or "").strip()
    if not emotional:
        return "neutral"
    # Strip parenthetical part for display (e.g. "joy (bright)" -> use "joy" for mapping)
    base = emotional.split("(")[0].strip() if "(" in emotional else emotional
    base_lower = base.lower()
    for key, value in EMOTION_MAP.items():
        if key in base_lower:
            return value
    return base or "neutral"


def _infer_key_mode(tech: Dict[str, Any]) -> str:
    """Infer key_mode from technical.key; only when key is missing or invalid."""
    key_raw = _get_str(tech, "key")
    if not key_raw:
        return DEFAULT_KEY_MODE
    parts = key_raw.split()
    root = parts[0] if parts else "C"
    mode = "major"
    if len(parts) > 1:
        mode_candidate = parts[1].lower()
        mode = mode_candidate if mode_candidate in VALID_MUSICAL_MODES else "major"
    return f"{root} {mode}"


def _infer_tempo(tech: Dict[str, Any]) -> int:
    """Infer tempo from technical.bpm; clamp to valid range."""
    bpm = tech.get("bpm") if isinstance(tech, dict) else getattr(tech, "bpm", None)
    if bpm is None:
        return DEFAULT_TEMPO
    try:
        bpm = int(bpm)
        return max(40, min(300, bpm))
    except (ValueError, TypeError):
        return DEFAULT_TEMPO


def _normalize_structure(tech: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Structure list for schema; infer only if missing or empty."""
    raw = tech.get("structure") if isinstance(tech, dict) else getattr(tech, "structure", None)
    if raw is None:
        return DEFAULT_STRUCTURE
    if isinstance(raw, list) and len(raw) == 0:
        return DEFAULT_STRUCTURE
    out: List[Dict[str, Any]] = []
    for item in raw:
        as_dict = pydantic_to_dict(item)
        if isinstance(as_dict, dict) and as_dict:
            out.append({
                "name": as_dict.get("name", "verse"),
                "bars": int(as_dict.get("bars", 4)),
                "repetitions": int(as_dict.get("repetitions", 1)),
            })
        else:
            out.append({"name": "verse", "bars": 4, "repetitions": 1})
    return out if out else DEFAULT_STRUCTURE


def _normalize_instruments(tech: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Instruments list for schema; infer only if missing or empty."""
    raw = tech.get("instruments") if isinstance(tech, dict) else getattr(tech, "instruments", None)
    if raw is None:
        return DEFAULT_INSTRUMENTS
    if isinstance(raw, list) and len(raw) == 0:
        return DEFAULT_INSTRUMENTS
    out: List[Dict[str, Any]] = []
    for item in raw:
        as_dict = pydantic_to_dict(item)
        if isinstance(as_dict, dict) and as_dict:
            inst = as_dict.get("instrument") or as_dict.get("name") or as_dict.get("type") or "piano"
            out.append({"instrument": str(inst), "techniques": as_dict.get("techniques", []) or []})
        else:
            out.append({"instrument": str(item), "techniques": []})
    return out if out else DEFAULT_INSTRUMENTS


def _timeline_present(tech: Dict[str, Any]) -> bool:
    raw = tech.get("timeline")
    if raw is None:
        return False
    if isinstance(raw, dict) and not raw:
        return False
    return True


def _orchestration_present(tech: Dict[str, Any]) -> bool:
    raw = tech.get("orchestration")
    if raw is None:
        return False
    raw = pydantic_to_dict(raw)
    if isinstance(raw, dict) and not raw.get("roles"):
        return False
    return True


def _structure_from_ttg(tech: Dict[str, Any]) -> List[Dict[str, Any]]:
    from music_brain.api_schemas.ttg_adapter import ttg_dict_to_structure_list

    timeline_raw = pydantic_to_dict(tech.get("timeline"))
    return ttg_dict_to_structure_list(timeline_raw)


def _instruments_from_orchestration(tech: Dict[str, Any]) -> List[Dict[str, Any]]:
    from music_brain.api_schemas.ttg_adapter import orchestration_dict_to_instruments

    orch = pydantic_to_dict(tech.get("orchestration"))
    return orchestration_dict_to_instruments(orch)


def _instruments_from_orchestration_gated(
    tech: Dict[str, Any], structure: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Energy-gated orchestration: per-section ``active_in:`` techniques (PRD §8.2)."""
    from music_brain.api_schemas.ttg_adapter import orchestration_with_energy_gating
    from music_brain.api_schemas.ttg_v1 import EnergyCurveV1, TTGOrchestrationV1

    orch = TTGOrchestrationV1.model_validate(pydantic_to_dict(tech.get("orchestration")))
    curve = EnergyCurveV1.model_validate(pydantic_to_dict(tech.get("energy_curve")))
    return orchestration_with_energy_gating(orch, energy_curve=curve, structure=structure)


class IntentPipeline:
    """
    Deterministic 3-stage pipeline: Normalize → Validate → Expand.

    Request shape (duck-typed): must have .intent with
    emotional_intent, core_wound, core_desire, technical,
    vulnerability_scale, narrative_arc. expand() also uses .intent.imagery_texture.
    """

    def run(self, request: Any) -> CompleteSongIntent:
        """Run full pipeline: normalize → validate → expand."""
        normalized = self.normalize(request)
        validated = self.validate(normalized)
        return self.expand(request, validated, normalized)

    # -------------------------------------------------------------------------
    # Stage 1: Normalize — only infer missing values; do not override explicit
    # -------------------------------------------------------------------------

    def normalize(self, request: Any) -> Dict[str, Any]:
        """
        Build intermediate dict from request for validation.

        - Applies emotion_map to emotional_intent to infer mood_primary (only when not explicitly provided).
        - Parses emotional string (e.g. strip parenthetical).
        - Derives key_mode, tempo, structure, instruments with fallbacks.
        - Does NOT set imagery_texture (Stage 3 only).
        """
        intent = request.intent
        emotional = _get_str(intent, "emotional_intent")
        tech = _tech_dict(request)

        # core_desire: use explicit if non-empty, else emotional_intent, else fallback
        core_desire = _get_str(intent, "core_desire")
        if not core_desire:
            core_desire = emotional or "unspecified"

        # mood_primary: infer from emotional_intent via emotion_map (Stage 1 only)
        mood_primary = _infer_mood_primary(emotional)

        # genre: explicit from tech or fallback
        genre = _get_str(tech, "genre")
        if not genre:
            genre = "pop"

        tempo = _infer_tempo(tech)
        key_mode = _infer_key_mode(tech)

        # PRD TTG v1: timeline + orchestration take precedence over flat fields.
        if _timeline_present(tech):
            structure = _structure_from_ttg(tech)
        else:
            structure = _normalize_structure(tech)

        if _orchestration_present(tech):
            if tech.get("energy_curve") is not None and structure:
                instruments = _instruments_from_orchestration_gated(tech, structure)
            else:
                instruments = _instruments_from_orchestration(tech)
        else:
            instruments = _normalize_instruments(tech)

        # Optional schema fields; use explicit if provided
        narrative_arc = _get_str(intent, "narrative_arc") or _get_str(tech, "narrative_arc") or DEFAULT_NARRATIVE_ARC
        groove_feel = _get_str(tech, "groove_feel") or DEFAULT_GROOVE_FEEL
        rule_to_break = _get_str(tech, "rule_to_break") or None
        rule_justification = _get_str(tech, "rule_justification") or None
        if rule_to_break == "":
            rule_to_break = None
        if rule_justification == "":
            rule_justification = None

        out = {
            "core_desire": core_desire,
            "mood_primary": mood_primary,
            "genre": genre,
            "tempo": tempo,
            "key_mode": key_mode,
            "structure": structure,
            "instruments": instruments,
            "allow_legacy_fallback": False,
            "groove_feel": groove_feel,
            "narrative_arc": narrative_arc,
            "rule_to_break": rule_to_break,
            "rule_justification": rule_justification,
        }
        energy = tech.get("energy_curve")
        if energy is not None:
            out["energy_curve"] = pydantic_to_dict(energy)
        return out

    # -------------------------------------------------------------------------
    # Stage 2: Validate — normalized dict → CompleteSongIntentRequest
    # -------------------------------------------------------------------------

    def validate(self, normalized: Dict[str, Any]) -> CompleteSongIntentRequest:
        """
        Convert normalized dict to CompleteSongIntentRequest.
        Pydantic handles validation and coercion.
        """
        structure = normalized.get("structure", DEFAULT_STRUCTURE)
        instruments = normalized.get("instruments", DEFAULT_INSTRUMENTS)
        payload = {k: normalized[k] for k in _ALLOWED_COMPLETE if k in normalized}
        payload["structure"] = structure
        payload["instruments"] = instruments
        try:
            if hasattr(CompleteSongIntentRequest, "model_validate"):
                return CompleteSongIntentRequest.model_validate(payload)
            return CompleteSongIntentRequest.parse_obj(payload)
        except Exception:
            raise

    # -------------------------------------------------------------------------
    # Stage 3: Expand — build CompleteSongIntent; imagery_texture ONLY here
    # -------------------------------------------------------------------------

    def expand(
        self,
        request: Any,
        validated: CompleteSongIntentRequest,
        normalized: Dict[str, Any] | None = None,
    ) -> CompleteSongIntent:
        """
        Build CompleteSongIntent from request + validated (+ optional normalized).

        - Precedence: validated > normalized > raw request. Never overwrite validated.
        - imagery_texture is set ONLY here, from request.intent.imagery_texture.
        - Populates song_root, song_intent, technical_constraints.
        """
        if normalized is None:
            normalized = self.normalize(request)
        norm = normalized or {}
        intent = request.intent

        # key_mode, tempo: validated is source of truth (schema always provides after validate)
        key_mode = getattr(validated, "key_mode", "") or norm.get("key_mode") or _get_str(_tech_dict(request), "key") or DEFAULT_KEY_MODE
        key_parts = key_mode.split()
        technical_key = key_parts[0] if key_parts else "C"
        technical_mode = key_parts[1].lower() if len(key_parts) > 1 else "major"
        tempo_val = getattr(validated, "tempo", None)
        if tempo_val is None:
            tempo_val = norm.get("tempo", DEFAULT_TEMPO)
        tempo_lo = min(140, max(60, tempo_val - 20))
        tempo_hi = min(140, max(60, tempo_val + 20))
        if tempo_hi < tempo_lo:
            tempo_hi = tempo_lo
        tempo_range = (
            tempo_lo,
            tempo_hi,
        )

        # core_event: request core_wound > validated.core_desire > normalized > request emotional
        core_event = _get_str(intent, "core_wound")
        if not core_event:
            core_event = getattr(validated, "core_desire", None) or norm.get("core_desire") or _get_str(intent, "emotional_intent") or ""

        # core_longing: validated > normalized > request
        core_longing = getattr(validated, "core_desire", None) or norm.get("core_desire") or _get_str(intent, "core_desire") or ""

        # mood_primary: validated > normalized > request emotional
        mood_primary = getattr(validated, "mood_primary", None) or norm.get("mood_primary") or _get_str(intent, "emotional_intent") or ""

        vulnerability_scale = getattr(intent, "vulnerability_scale", None)
        if vulnerability_scale is None:
            vulnerability_scale = 0.5
        else:
            try:
                vulnerability_scale = float(vulnerability_scale)
                vulnerability_scale = max(0.0, min(1.0, vulnerability_scale))
            except (ValueError, TypeError):
                vulnerability_scale = 0.5

        # imagery_texture: ONLY set here (Stage 3), from request.intent only
        imagery_texture = getattr(intent, "imagery_texture", "") or ""

        # narrative_arc: validated > normalized > request
        narrative_arc = getattr(validated, "narrative_arc", None) or norm.get("narrative_arc") or _get_str(intent, "narrative_arc") or ""

        # technical_*: validated first, then normalized, then request
        technical_genre = getattr(validated, "genre", None) or norm.get("genre") or _get_str(_tech_dict(request), "genre") or ""
        technical_groove_feel = getattr(validated, "groove_feel", None) or norm.get("groove_feel") or _get_str(_tech_dict(request), "groove_feel") or ""
        technical_rule_to_break = getattr(validated, "rule_to_break", None) or norm.get("rule_to_break") or _get_str(_tech_dict(request), "rule_to_break") or ""
        rule_breaking_justification = getattr(validated, "rule_justification", None) or norm.get("rule_justification") or _get_str(_tech_dict(request), "rule_justification") or ""

        return CompleteSongIntent(
            core_event=core_event,
            core_longing=core_longing,
            mood_primary=mood_primary,
            imagery_texture=imagery_texture,
            narrative_arc=narrative_arc,
            vulnerability_scale=vulnerability_scale,
            technical_genre=technical_genre,
            technical_tempo_range=tempo_range,
            technical_key=technical_key,
            technical_mode=technical_mode,
            technical_groove_feel=technical_groove_feel,
            technical_rule_to_break=technical_rule_to_break or "",
            rule_breaking_justification=rule_breaking_justification or "",
            created=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
