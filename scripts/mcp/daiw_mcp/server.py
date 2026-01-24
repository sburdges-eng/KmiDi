"""
DAiW Music-Brain MCP Server

Unified server that registers all 24+ MCP tools for music production.
"""

import asyncio
import json
import sys
from typing import Any, Dict, List, Optional

try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    Server = None
    Tool = None
    TextContent = None

from daiw_mcp.tools.harmony import register_tools as register_harmony
from daiw_mcp.tools.groove import register_tools as register_groove
from daiw_mcp.tools.intent import register_tools as register_intent
from daiw_mcp.tools.audio_analysis import register_tools as register_audio
from daiw_mcp.tools.teaching import register_tools as register_teaching


# Store tool handlers from each module
_module_servers: Dict[str, Server] = {}


def create_server() -> Optional[Server]:
    """
    Create and configure MCP server with all tools.
    
    Returns:
        Configured MCP Server instance, or None if MCP is not available
    """
    if not MCP_AVAILABLE:
        return None
    
    # Create individual servers for each module to collect their tools
    harmony_server = Server("harmony")
    groove_server = Server("groove")
    intent_server = Server("intent")
    audio_server = Server("audio")
    teaching_server = Server("teaching")
    
    register_harmony(harmony_server)
    register_groove(groove_server)
    register_intent(intent_server)
    register_audio(audio_server)
    register_teaching(teaching_server)
    
    # Store module servers for routing
    _module_servers["harmony"] = harmony_server
    _module_servers["groove"] = groove_server
    _module_servers["intent"] = intent_server
    _module_servers["audio"] = audio_server
    _module_servers["teaching"] = teaching_server
    
    # Create unified server
    unified_server = Server("daiw-music-brain")
    
    # Aggregate all tools
    @unified_server.list_tools()
    async def list_tools() -> List[Tool]:
        """List all available tools from all modules."""
        all_tools = []
        for server in _module_servers.values():
            tools = await server.list_tools()
            all_tools.extend(tools)
        return all_tools
    
    # Route tool calls to appropriate module
    @unified_server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Route tool calls to the appropriate module handler."""
        # Try each module server
        for module_name, server in _module_servers.items():
            tools = await server.list_tools()
            tool_names = [t.name for t in tools]
            if name in tool_names:
                # This module handles this tool
                return await server.call_tool(name, arguments)
        
        # Tool not found
        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}, indent=2))]
    
    return unified_server


async def run_server_stdio():
    """
    Run the MCP server with stdio transport (manual implementation).
    
    This implements the MCP protocol for tool invocation via stdin/stdout.
    """
    if not MCP_AVAILABLE:
        print("Error: MCP library not available. Install with: pip install mcp", file=sys.stderr)
        sys.exit(1)
    
    server = create_server()
    if server is None:
        print("Error: Failed to create MCP server", file=sys.stderr)
        sys.exit(1)
    
    # Manual stdio implementation
    print("DAiW Music-Brain MCP Server starting...", file=sys.stderr)
    
    while True:
        try:
            line = await asyncio.get_event_loop().run_in_executor(
                None, sys.stdin.readline
            )
            if not line:
                break
            
            line = line.strip()
            if not line:
                continue
            
            request = json.loads(line)
            method = request.get("method", "")
            params = request.get("params", {})
            req_id = request.get("id")
            
            response = {"jsonrpc": "2.0", "id": req_id}
            
            if method == "initialize":
                response["result"] = {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {},
                    },
                    "serverInfo": {
                        "name": "daiw-music-brain",
                        "version": "1.0.0",
                    },
                }
            
            elif method == "tools/list":
                tools = await server.list_tools()
                # Convert Tool objects to dicts
                tools_dict = [
                    {
                        "name": t.name,
                        "description": t.description,
                        "inputSchema": t.inputSchema,
                    }
                    for t in tools
                ]
                response["result"] = {"tools": tools_dict}
            
            elif method == "tools/call":
                tool_name = params.get("name", "")
                tool_args = params.get("arguments", {})
                result = await server.call_tool(tool_name, tool_args)
                # Convert TextContent to dict
                content = [
                    {"type": c.type, "text": c.text}
                    for c in result
                ]
                response["result"] = {"content": content}
            
            elif method == "notifications/initialized":
                # No response needed for notifications
                continue
            
            else:
                response["error"] = {
                    "code": -32601,
                    "message": f"Method not found: {method}",
                }
            
            # Write response to stdout
            print(json.dumps(response), flush=True)
        
        except json.JSONDecodeError as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32700,
                    "message": f"Parse error: {str(e)}"
                }
            }
            print(json.dumps(error_response), flush=True)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            error_response = {
                "jsonrpc": "2.0",
                "id": req_id if 'req_id' in locals() else None,
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }
            print(json.dumps(error_response), flush=True)


def main():
    """CLI entry point for running the MCP server."""
    try:
        asyncio.run(run_server_stdio())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error running MCP server: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
