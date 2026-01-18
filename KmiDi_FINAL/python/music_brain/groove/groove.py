"""
Groove-related MCP tools.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict

from .common import api, make_midi_payload, midi_file_context, run_sync

JsonDict = Dict[str, Any]


async def extract_groove_tool(
    midi_base64: str = "",
    midi_path: str = "",
) -> JsonDict:
    """
    Extract a groove profile from a MIDI file.
    """

    with midi_file_context(
        midi_path=midi_path or None,
        midi_base64=midi_base64 or None,
    ) as source_path:
        groove = await run_sync(api.extract_groove_from_midi, source_path)
    return {"groove": groove}


async def apply_groove_tool(
    midi_base64: str = "",
    midi_path: str = "",
    filename: str = "source.mid",
    genre: str = "funk",
    intensity: float = 0.5,
) -> JsonDict:
    """
    Apply a genre groove template to a MIDI file and return the modified file.
    """

    with midi_file_context(
        midi_path=midi_path or None,
        midi_base64=midi_base64 or None,
    ) as source_path, tempfile.TemporaryDirectory(prefix="daiw_mcp_") as tmpdir:
        output_path = os.path.join(tmpdir, "grooved.mid")
        result_path = await run_sync(
            api.apply_groove_to_midi,
            source_path,
            genre,
            intensity,
            output_path,
        )

        return {
            "genre": genre,
            "intensity": intensity,
            "midi": make_midi_payload(result_path, filename=f"{os.path.splitext(filename)[0]}_grooved.mid"),
        }


async def humanize_midi_tool(
    midi_base64: str = "",
    midi_path: str = "",
    filename: str = "source.mid",
    complexity: float = 0.5,
    vulnerability: float = 0.5,
    preset: str | None = None,
    drum_channel: int = 9,
    enable_ghost_notes: bool = True,
) -> JsonDict:
    """
    Apply the drum humanization engine and return the resulting MIDI.
    """

    with midi_file_context(
        midi_path=midi_path or None,
        midi_base64=midi_base64 or None,
    ) as source_path, tempfile.TemporaryDirectory(prefix="daiw_mcp_") as tmpdir:
        output_path = os.path.join(tmpdir, "humanized.mid")
        result = await run_sync(
            api.humanize_drums,
            source_path,
            complexity,
            vulnerability,
            preset,
            drum_channel,
            enable_ghost_notes,
            output_path,
        )

        return {
            **result,
            "midi": make_midi_payload(result["output_path"], filename=f"{os.path.splitext(filename)[0]}_humanized.mid"),
        }


def register_tools(server) -> None:
    """Register groove tools with FastMCP."""

    server.tool(
        name="daiw.extract_groove",
        description="Extract timing/velocity groove information from a MIDI file.",
    )(extract_groove_tool)

    server.tool(
        name="daiw.apply_groove",
        description="Apply a DAiW genre groove template to a MIDI file.",
    )(apply_groove_tool)

    server.tool(
        name="daiw.humanize_midi",
        description="Apply the Drunken Drummer humanization engine to drum MIDI.",
    )(humanize_midi_tool)


__all__ = ["register_tools"]

