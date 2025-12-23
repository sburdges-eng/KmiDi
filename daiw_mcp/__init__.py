"""
DAiW Music-Brain MCP Server Package

Unified MCP server for music production tools.
"""

__version__ = "1.0.0"

from daiw_mcp.server import create_server, run_server

__all__ = [
    "create_server",
    "run_server",
]
