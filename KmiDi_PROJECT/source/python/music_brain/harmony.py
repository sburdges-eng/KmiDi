"""
Harmony-related MCP tools.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, Optional

from data.harmony_generator import (
    HarmonyGenerator,
    HarmonyResult,
    generate_midi_from_harmony,
)
from .common import (
    api,
    make_midi_payload,
    midi_file_context,
    parse_intent_json,
    run_sync,
)

JsonDict = Dict[str, Any]


async def analyze_progression_tool(progression: str) -> JsonDict:
    """
    Diagnose a chord progression string and report potential issues.
    """

    diagnosis = await run_sync(api.diagnose_progression, progression)
    return diagnosis


async def diagnose_chords_tool(progression: str) -> JsonDict:
    """
    Highlight only the issues/suggestions for a progression.
    """

    diagnosis = await run_sync(api.diagnose_progression, progression)
    return {
        "key": diagnosis.get("key"),
        "mode": diagnosis.get("mode"),
        "issues": diagnosis.get("issues", []),
        "suggestions": diagnosis.get("suggestions", []),
    }


async def analyze_chords_tool(
    midi_base64: str = "",
    midi_path: str = "",
    include_sections: bool = False,
) -> JsonDict:
    """
    Analyze chords from a MIDI file, optionally returning section data.
    """

    with midi_file_context(
        midi_path=midi_path or None,
        midi_base64=midi_base64 or None,
    ) as source_path:
        analysis = await run_sync(
            api.analyze_midi_chords,
            source_path,
            include_sections,
        )
    return analysis


async def generate_harmony_tool(
    intent_json: Optional[str] = None,
    key: str = "C",
    mode: str = "major",
    pattern: str = "I-V-vi-IV",
    tempo_bpm: int = 82,
    include_midi: bool = True,
) -> JsonDict:
    """
    Generate harmony from either a CompleteSongIntent or basic parameters.
    """

    payload: JsonDict
    if intent_json:
        intent = parse_intent_json(intent_json)
        with tempfile.TemporaryDirectory(prefix="daiw_mcp_") as tmpdir:
            midi_path = f"{tmpdir}/harmony.mid"
            result = await run_sync(
                api.generate_harmony_from_intent,
                intent,
                midi_path if include_midi else None,
                tempo_bpm,
            )
            payload = result
            if include_midi and "midi_path" in result:
                payload["midi"] = make_midi_payload(result["midi_path"], filename="intent_harmony.mid")
    else:
        with tempfile.TemporaryDirectory(prefix="daiw_mcp_") as tmpdir:
            midi_path = f"{tmpdir}/basic.mid"
            result = await run_sync(
                api.generate_basic_progression,
                key,
                mode,
                pattern,
                midi_path if include_midi else None,
                tempo_bpm,
            )
            payload = result
            if include_midi and "midi_path" in result:
                payload["midi"] = make_midi_payload(result["midi_path"], filename="basic_progression.mid")

    return payload


async def suggest_reharmonization_tool(
    progression: str,
    style: str = "jazz",
    count: int = 3,
) -> JsonDict:
    """
    Suggest alternative reharmonizations for a chord progression.
    """

    suggestions = await run_sync(api.suggest_reharmonizations, progression, style, count)
    return {"suggestions": suggestions}


async def therapy_harmony_snapshot_tool(
    text: str,
    motivation: int = 7,
    chaos_tolerance: float = 0.5,
) -> JsonDict:
    """
    Short-cut helper that runs the therapy session and surfaces the harmonic plan.
    """

    session = await run_sync(api.therapy_session, text, motivation, chaos_tolerance)
    return session


def register_tools(server) -> None:
    """
    Register harmony tools with the provided FastMCP server.
    """

    server.tool(
        name="daiw.analyze_chords",
        description="Analyze chords (and optionally sections) from a MIDI file.",
    )(analyze_chords_tool)

    server.tool(
        name="daiw.analyze_progression",
        description="Diagnose a chord progression and surface emotional notes.",
    )(analyze_progression_tool)

    server.tool(
        name="daiw.diagnose_chords",
        description="Focus on issues/suggestions for a chord progression string.",
    )(diagnose_chords_tool)

    server.tool(
        name="daiw.generate_harmony",
        description="Generate harmony from an intent JSON blob or basic parameters.",
    )(generate_harmony_tool)

    server.tool(
        name="daiw.suggest_reharmonization",
        description="Suggest alternative progressions in a requested style.",
    )(suggest_reharmonization_tool)

    server.tool(
        name="daiw.therapy_harmony_snapshot",
        description="Run the therapy engine and return the harmonic plan snapshot.",
    )(therapy_harmony_snapshot_tool)


__all__ = [
    "HarmonyGenerator",
    "HarmonyResult",
    "generate_midi_from_harmony",
    "register_tools",
]

