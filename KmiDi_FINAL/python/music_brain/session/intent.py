"""
Intent-schema MCP tools.
"""

from __future__ import annotations

from typing import Any, Dict

from music_brain.session.intent_schema import (
    CompleteSongIntent,
    SongIntent,
    SongRoot,
    SystemDirective,
    TechnicalConstraints,
)

from .common import api, parse_intent_json, run_sync

JsonDict = Dict[str, Any]


def _default_intent(title: str) -> CompleteSongIntent:
    return CompleteSongIntent(
        title=title,
        song_root=SongRoot(
            core_event="[What happened?]",
            core_resistance="[What holds you back?]",
            core_longing="[What do you want to feel?]",
            core_stakes="Personal",
            core_transformation="[How should you feel at the end?]",
        ),
        song_intent=SongIntent(
            mood_primary="[Primary emotion]",
            mood_secondary_tension=0.5,
            imagery_texture="[Visual/tactile quality]",
            vulnerability_scale="Medium",
            narrative_arc="Climb-to-Climax",
        ),
        technical_constraints=TechnicalConstraints(
            technical_genre="[Genre]",
            technical_tempo_range=(80, 120),
            technical_key="F",
            technical_mode="major",
            technical_groove_feel="Organic/Breathing",
            technical_rule_to_break="",
            rule_breaking_justification="",
        ),
        system_directive=SystemDirective(
            output_target="Chord progression",
            output_feedback_loop="Harmony",
        ),
    )


async def create_intent_tool(title: str = "Untitled Song") -> JsonDict:
    """Return a template CompleteSongIntent."""

    intent = _default_intent(title)
    return intent.to_dict()


async def process_intent_tool(intent_json: str) -> JsonDict:
    """
    Process an intent file and return generated musical elements.
    """

    intent = parse_intent_json(intent_json)
    result = await run_sync(api.process_song_intent, intent)
    return result


async def validate_intent_tool(intent_json: str) -> JsonDict:
    """
    Validate an intent file and list any issues.
    """

    intent = parse_intent_json(intent_json)
    issues = await run_sync(api.validate_song_intent, intent)
    return {"valid": len(issues) == 0, "issues": issues}


async def suggest_rulebreaks_tool(emotion: str) -> JsonDict:
    """Suggest rules to break for a target emotion."""

    suggestions = await run_sync(api.suggest_rule_breaks, emotion)
    return {"emotion": emotion, "suggestions": suggestions}


def register_tools(server) -> None:
    """Register intent-related tools."""

    server.tool(
        name="daiw.intent.create_template",
        description="Create a blank CompleteSongIntent template.",
    )(create_intent_tool)

    server.tool(
        name="daiw.intent.process",
        description="Process a CompleteSongIntent file into musical directives.",
    )(process_intent_tool)

    server.tool(
        name="daiw.intent.validate",
        description="Validate a CompleteSongIntent file.",
    )(validate_intent_tool)

    server.tool(
        name="daiw.intent.suggest_rulebreaks",
        description="Suggest rules to break for a target emotion.",
    )(suggest_rulebreaks_tool)


__all__ = ["register_tools"]

