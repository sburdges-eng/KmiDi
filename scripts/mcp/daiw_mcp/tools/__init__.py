"""
DAiW MCP Tools Package

Exports all tool registration functions.
"""

from daiw_mcp.tools.harmony import register_tools as register_harmony_tools
from daiw_mcp.tools.groove import register_tools as register_groove_tools
from daiw_mcp.tools.intent import register_tools as register_intent_tools
from daiw_mcp.tools.audio_analysis import register_tools as register_audio_tools
from daiw_mcp.tools.teaching import register_tools as register_teaching_tools

__all__ = [
    "register_harmony_tools",
    "register_groove_tools",
    "register_intent_tools",
    "register_audio_tools",
    "register_teaching_tools",
]
