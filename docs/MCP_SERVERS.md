# MCP Servers Documentation

**Date**: 2026-01-21  
**Status**: Complete MCP Server Catalog

This document catalogs all Model Context Protocol (MCP) servers in the KmiDi project, their configuration, and usage.

## Overview

KmiDi includes multiple MCP servers that provide AI agent capabilities for music generation, workflow orchestration, and development tools. All servers follow the Model Context Protocol specification.

## Available MCP Servers

### 1. mcp_penta_swarm

**Location**: `mcp_penta_swarm/`  
**Purpose**: Swarm orchestration for AI agents working on music generation tasks

**Files**:
- `server.py` - Main MCP server implementation
- `__main__.py` - Entry point for running as module
- `README.md` - Server documentation
- `requirements.txt` - Python dependencies
- `.env.example` - Environment variable template

**Configuration**:
- Copy `.env.example` to `.env` and configure as needed
- Set environment variables for API keys, model endpoints, etc.

**Usage**:
```bash
# Run as module
python -m mcp_penta_swarm

# Or directly
python mcp_penta_swarm/server.py
```

**Dependencies**:
- music_brain modules
- penta_core modules
- MCP SDK

**Tools Provided**:
- Swarm orchestration
- Multi-agent coordination
- Task distribution

### 2. mcp_workstation

**Location**: `mcp_workstation/`  
**Purpose**: Development workflow orchestration and planning

**Files**:
- `server.py` - Main MCP server
- `orchestrator.py` - Workflow orchestration logic
- `cpp_planner.py` - C++ code planning tools
- `phases.py` - Development phase management
- `proposals.py` - Proposal generation
- `models.py` - Data models
- `cli.py` - Command-line interface
- `debug.py` - Debugging utilities
- `configs/` - Configuration files
  - `claude_desktop.json` - Claude Desktop configuration
  - `cursor.json` - Cursor IDE configuration
  - `setup_guide.md` - Setup instructions

**Configuration**:
- Configure in `configs/cursor.json` or `configs/claude_desktop.json`
- Follow `configs/setup_guide.md` for setup instructions

**Usage**:
```bash
# Run server
python -m mcp_workstation

# Or use CLI
python mcp_workstation/cli.py
```

**Dependencies**:
- music_brain modules
- penta_core modules
- MCP SDK

**Tools Provided**:
- Workflow orchestration
- C++ code planning
- Development phase management
- Proposal generation

### 3. daiw_mcp

**Location**: `daiw_mcp/`  
**Purpose**: DAiW (Digital Audio Intelligence Workstation) tool integrations

**Files**:
- `server.py` - Main MCP server
- `README.md` - Documentation
- `tools/` - Tool implementations
  - `audio_analysis.py` - Audio analysis tools
  - `groove.py` - Groove processing tools
  - `harmony.py` - Harmony generation tools
  - `intent.py` - Intent processing tools
  - `teaching.py` - Teaching system tools
  - `_server_utils.py` - Server utilities
- `tests/` - Test suite
  - `test_mcp_tools.py` - Tool tests
  - `conftest.py` - Test configuration

**Configuration**:
- No special configuration required
- Uses music_brain modules directly

**Usage**:
```bash
# Run server
python -m daiw_mcp

# Run tests
python -m pytest daiw_mcp/tests/
```

**Dependencies**:
- music_brain modules
- MCP SDK

**Tools Provided**:
- Audio analysis
- Groove extraction and application
- Harmony generation
- Intent processing
- Teaching system integration

### 4. mcp_todo

**Location**: `mcp_todo/`  
**Purpose**: Task management integration

**Files**:
- `server.py` - Main MCP server
- `http_server.py` - HTTP interface
- `__init__.py` - Package initialization

**Configuration**:
- Configure task storage location
- Set up HTTP server port if using HTTP interface

**Usage**:
```bash
# Run MCP server
python -m mcp_todo

# Or run HTTP server
python mcp_todo/http_server.py
```

**Dependencies**:
- music_brain modules (for context)
- MCP SDK

**Tools Provided**:
- Task creation
- Task management
- Task querying
- Task completion tracking

## MCP Server Configuration

### Claude Desktop Configuration

Add to Claude Desktop's MCP configuration file (typically `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "mcp_penta_swarm": {
      "command": "python",
      "args": ["-m", "mcp_penta_swarm"],
      "cwd": "/path/to/KmiDi"
    },
    "mcp_workstation": {
      "command": "python",
      "args": ["-m", "mcp_workstation"],
      "cwd": "/path/to/KmiDi"
    },
    "daiw_mcp": {
      "command": "python",
      "args": ["-m", "daiw_mcp"],
      "cwd": "/path/to/KmiDi"
    },
    "mcp_todo": {
      "command": "python",
      "args": ["-m", "mcp_todo"],
      "cwd": "/path/to/KmiDi"
    }
  }
}
```

### Cursor IDE Configuration

Add to Cursor's MCP configuration (typically in workspace settings):

```json
{
  "mcp": {
    "servers": {
      "mcp_penta_swarm": {
        "command": "python",
        "args": ["-m", "mcp_penta_swarm"]
      },
      "mcp_workstation": {
        "command": "python",
        "args": ["-m", "mcp_workstation"]
      },
      "daiw_mcp": {
        "command": "python",
        "args": ["-m", "daiw_mcp"]
      },
      "mcp_todo": {
        "command": "python",
        "args": ["-m", "mcp_todo"]
      }
    }
  }
}
```

## Verification

### Testing MCP Servers

Each server can be tested independently:

```bash
# Test mcp_penta_swarm
python -m mcp_penta_swarm --test

# Test mcp_workstation
python -m mcp_workstation --test

# Test daiw_mcp
python -m pytest daiw_mcp/tests/

# Test mcp_todo
python -m mcp_todo --test
```

### Health Checks

All servers should respond to health check requests:

```python
# Example health check
import requests
response = requests.get("http://localhost:8000/health")  # If HTTP interface available
```

## Dependencies

### Common Dependencies

All MCP servers require:
- Python 3.9+
- MCP SDK (mcp package)
- music_brain modules
- penta_core modules (for some servers)

### Installation

```bash
# Install MCP SDK
pip install mcp

# Install server-specific dependencies
pip install -r mcp_penta_swarm/requirements.txt
pip install -r mcp_workstation/requirements.txt  # If available
```

## Troubleshooting

### Server Won't Start

1. Check Python version: `python --version` (should be 3.9+)
2. Verify dependencies: `pip list | grep mcp`
3. Check configuration files
4. Review server logs

### Connection Issues

1. Verify MCP client configuration
2. Check server is running: `ps aux | grep mcp`
3. Verify port availability (if using HTTP)
4. Check firewall settings

### Tool Not Available

1. Verify tool is registered in server
2. Check server logs for errors
3. Verify dependencies are installed
4. Check tool implementation file exists

## Integration with Development Workflow

### Using MCP Servers in Development

MCP servers can be used during development to:
- Generate code plans
- Orchestrate workflows
- Analyze audio
- Process intents
- Manage tasks

### Starting Servers for Development

```bash
# Start all MCP servers (if script available)
./scripts/start-mcp-servers.sh

# Or start individually
python -m mcp_workstation &
python -m daiw_mcp &
```

## References

- MCP Specification: [Model Context Protocol](https://modelcontextprotocol.io)
- System Architecture: `docs/SYSTEM_ARCHITECTURE.md`
- System Inventory: `docs/SYSTEM_INVENTORY.md`
