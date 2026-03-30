# DAiW Music-Brain MCP Server

Unified Model Context Protocol (MCP) server providing 24+ tools for music production, analysis, and composition.

## Overview

The DAiW MCP server exposes music production intelligence tools through the Model Context Protocol, enabling AI assistants (Claude, ChatGPT, Gemini, etc.) to interact with the DAiW Music-Brain toolkit.

## Installation

```bash
# Install dependencies
pip install mcp

# The server is part of the music_brain package
pip install -e .
```

## Usage

### Running the Server

```bash
# Run as a module
python -m daiw_mcp.server

# Or use the CLI entry point (if configured)
daiw-mcp-server
```

### MCP Client Configuration

Configure the server in your MCP client (e.g., Claude Desktop, Cursor):

```json
{
  "mcpServers": {
    "daiw-music-brain": {
      "command": "python",
      "args": ["-m", "daiw_mcp.server"]
    }
  }
}
```

## Available Tools

### Harmony Tools (6 tools)

- **analyze_progression** - Analyze chord progression for harmonic characteristics, emotional character, and rule breaks
- **generate_harmony** - Generate harmony from emotional intent or basic parameters
- **diagnose_chords** - Diagnose harmonic issues in a chord progression
- **suggest_reharmonization** - Suggest reharmonization alternatives
- **find_key** - Detect the key of a chord progression
- **voice_leading** - Optimize voice leading for a chord progression

### Groove Tools (5 tools)

- **extract_groove** - Extract groove characteristics (timing, velocity, swing) from MIDI
- **apply_groove** - Apply a genre groove template to MIDI
- **analyze_pocket** - Analyze the timing pocket (groove feel) of MIDI
- **humanize_midi** - Add human feel with complexity and vulnerability parameters
- **quantize_smart** - Smart quantization that preserves musical feel

### Intent Tools (4 tools)

- **create_intent** - Create a new song intent template with three-phase schema
- **process_intent** - Process CompleteSongIntent to generate musical elements
- **validate_intent** - Validate a CompleteSongIntent file against the schema
- **suggest_rulebreaks** - Suggest intentional rule-breaks based on emotion

### Audio Analysis Tools (6 tools)

- **detect_bpm** - Detect tempo (BPM) from an audio file
- **detect_key** - Detect musical key and mode from audio
- **analyze_audio_feel** - Analyze groove feel and energy characteristics
- **extract_chords** - Extract chord progression from audio
- **detect_scale** - Detect scales/modes from audio
- **analyze_theory** - Complete music theory analysis (scales, modes, harmonic complexity)

### Teaching Tools (3 tools)

- **explain_rulebreak** - Explain a rule-breaking technique with examples
- **get_progression_info** - Get detailed information about a chord progression
- **emotion_to_music** - Map an emotion to musical parameters

**Total: 24 tools**

## Examples

### Example 1: Analyze a Chord Progression

```python
# Via MCP client
{
  "name": "analyze_progression",
  "arguments": {
    "progression": "F-C-Dm-Bbm",
    "key": "F major"
  }
}
```

### Example 2: Generate Harmony from Intent

```python
{
  "name": "generate_harmony",
  "arguments": {
    "emotion": "nostalgia",
    "key": "F",
    "mode": "major",
    "output_midi": "output.mid"
  }
}
```

### Example 3: Detect BPM from Audio

```python
{
  "name": "detect_bpm",
  "arguments": {
    "audio_file": "song.wav"
  }
}
```

## Architecture

```
daiw_mcp/
├── __init__.py              # Package exports
├── server.py                # Unified MCP server
├── tools/
│   ├── __init__.py          # Tool exports
│   ├── harmony.py           # 6 harmony tools
│   ├── groove.py            # 5 groove tools
│   ├── intent.py            # 4 intent tools
│   ├── audio_analysis.py    # 6 audio tools
│   └── teaching.py          # 3 teaching tools
└── tests/
    └── test_mcp_tools.py    # Comprehensive tests
```

## Development

### Running Tests

```bash
pytest daiw_mcp/tests/test_mcp_tools.py -v
```

### Adding New Tools

1. Create tool function in appropriate module (`daiw_mcp/tools/`)
2. Register tool in `register_tools()` function
3. Add test in `daiw_mcp/tests/test_mcp_tools.py`
4. Update this README

## Dependencies

- `mcp` - Model Context Protocol library
- `music_brain` - DAiW Music-Brain core library
- `librosa` - Audio analysis (for audio tools)
- `mido` - MIDI processing (for groove tools)

## License

See main project LICENSE file.

## Related Documentation

- [DEVELOPMENT_ROADMAP_music-brain.md](../DEVELOPMENT_ROADMAP_music-brain.md)
- [CLAUDE.md](../CLAUDE.md) - Main project documentation
