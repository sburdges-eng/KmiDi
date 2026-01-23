# Penta-Core Server

An integration service that aggregates the top 5 AI platforms into a single unified toolset for IDE integration (Cursor/VS Code).

## Overview

The Penta-Core server provides 5 specialized AI tools, each powered by a different leading AI platform:

1. **consult_architect** (OpenAI GPT-4o) - High-level system design and architecture patterns
2. **consult_developer** (Anthropic Claude 3.5 Sonnet) - Clean code, refactoring, and best practices
3. **consult_researcher** (Google Gemini 1.5 Pro) - Deep context analysis with massive context window
4. **consult_maverick** (xAI Grok Beta) - Red teaming and unconventional solutions
5. **fetch_repo_context** (GitHub API) - Fetch file content/tree from GitHub repositories

## Installation

### From the KmiDi repository root:

```bash
# Install with MCP dependencies
pip install -e ".[mcp]"
```

### Configure API Keys

Copy the `.env.example` file to `.env` and fill in your API keys:

```bash
cp mcp_penta_swarm/.env.example mcp_penta_swarm/.env
```

Required API keys:
- `OPENAI_API_KEY` - For the Architect (GPT-4o)
- `ANTHROPIC_API_KEY` - For the Developer (Claude 3.5 Sonnet)
- `GOOGLE_API_KEY` - For the Researcher (Gemini 1.5 Pro)
- `XAI_API_KEY` - For the Maverick (Grok Beta)
- `GITHUB_TOKEN` - For fetching repository context

## Usage

### Running the Server

```bash
# As a module
python -m mcp_penta_swarm

# Or via the installed script
mcp-penta-swarm
```

### Using in Python Code

```python
from mcp_penta_swarm.server import (
    consult_architect,
    consult_developer,
    consult_researcher,
    consult_maverick,
    fetch_repo_context
)

# Get architectural guidance
response = await consult_architect(
    "Design a lock-free ring buffer for real-time audio processing"
)

# Get production-ready code
code = await consult_developer(
    "Refactor this function to handle edge cases: ..."
)

# Deep analysis with large context
analysis = await consult_researcher(
    prompt="Find potential issues in this codebase",
    context="<large amount of code/docs here>"
)

# Get creative criticism
critique = await consult_maverick(
    "Here's my plan for the new feature: ..."
)

# Fetch context from GitHub
content = await fetch_repo_context(
    owner="sburdges-eng",
    repo="KmiDi",
    path="README.md"
)
```

## Tool Descriptions

### consult_architect (OpenAI GPT-4o)

**Use for:**
- High-level system design
- Class structure planning
- Design pattern recommendations
- Architecture reviews

**System Prompt:** "You are a Systems Architect. Focus on high-level logic, class structure, and design patterns."

### consult_developer (Anthropic Claude 3.5 Sonnet)

**Use for:**
- Writing production-ready code
- Code refactoring
- Security reviews
- Best practices guidance

**System Prompt:** "You are a Senior Engineer. Focus on clean code, safety, and refactoring. Output production-ready code."

### consult_researcher (Google Gemini 1.5 Pro)

**Use for:**
- Analyzing large amounts of documentation
- Finding edge cases
- Deep context analysis
- Research synthesis

**System Prompt:** "You are a Lead Researcher. Analyze documentation and find edge cases with your massive context window."

### consult_maverick (xAI Grok Beta)

**Use for:**
- Red teaming your plans
- Finding non-obvious flaws
- Getting unconventional solutions
- Challenging assumptions

**System Prompt:** "You are a Maverick Engineer. Criticize the plan, find non-obvious flaws, and suggest lateral/unconventional solutions. Be direct."

### fetch_repo_context (GitHub API)

**Use for:**
- Getting context from external repositories
- Fetching specific files for analysis
- Listing directory contents

**Returns:** File content or directory listing as JSON

## Architecture

The server uses `fastmcp` for the MCP server framework and implements lazy loading for AI clients to minimize startup time and resource usage.

Each tool is implemented as an async function decorated with `@mcp.tool()`, making them available through the MCP protocol for IDE integration.

## Error Handling

All tools include robust error handling:
- Configuration errors (missing API keys)
- API rate limiting
- Network timeouts
- Invalid responses

Errors are returned as descriptive strings rather than raising exceptions, allowing the IDE to display them to users.

## Contributing

When adding new tools or modifying existing ones:

1. Follow the existing pattern of lazy client initialization
2. Include comprehensive docstrings
3. Add proper error handling
4. Keep system prompts focused and specific
5. Test with various inputs before committing

## License

MIT License - See LICENSE file in the repository root.
