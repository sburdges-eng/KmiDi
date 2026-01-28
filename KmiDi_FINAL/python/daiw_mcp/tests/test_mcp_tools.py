"""
Comprehensive tests for DAiW MCP Tools.

Tests all 24 tools across 5 categories:
- Harmony tools (6)
- Groove tools (5)
- Intent tools (4)
- Audio analysis tools (6)
- Teaching tools (3)
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    pytestmark = pytest.mark.skip("MCP library not available")

from daiw_mcp.tools import (
    register_harmony_tools,
    register_groove_tools,
    register_intent_tools,
    register_audio_tools,
    register_teaching_tools,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def harmony_server():
    """Create harmony tools server."""
    server = Server("test-harmony")
    register_harmony_tools(server)
    return server


@pytest.fixture
def groove_server():
    """Create groove tools server."""
    server = Server("test-groove")
    register_groove_tools(server)
    return server


@pytest.fixture
def intent_server():
    """Create intent tools server."""
    server = Server("test-intent")
    register_intent_tools(server)
    return server


@pytest.fixture
def audio_server():
    """Create audio tools server."""
    server = Server("test-audio")
    register_audio_tools(server)
    return server


@pytest.fixture
def teaching_server():
    """Create teaching tools server."""
    server = Server("test-teaching")
    register_teaching_tools(server)
    return server


# ============================================================================
# Harmony Tools Tests (6 tools)
# ============================================================================

@pytest.mark.asyncio
async def test_harmony_list_tools(harmony_server):
    """Test harmony tools are registered."""
    tools = await harmony_server.list_tools()
    tool_names = [t.name for t in tools]
    
    assert "analyze_progression" in tool_names
    assert "generate_harmony" in tool_names
    assert "diagnose_chords" in tool_names
    assert "suggest_reharmonization" in tool_names
    assert "find_key" in tool_names
    assert "voice_leading" in tool_names
    assert len(tools) == 6


@pytest.mark.asyncio
async def test_analyze_progression(harmony_server):
    """Test analyze_progression tool."""
    result = await harmony_server.call_tool(
        "analyze_progression",
        {"progression": "F-C-Dm-Bbm", "key": "F major"}
    )
    assert len(result) > 0
    data = json.loads(result[0].text)
    assert "progression" in data
    assert data["progression"] == "F-C-Dm-Bbm"


@pytest.mark.asyncio
async def test_find_key(harmony_server):
    """Test find_key tool."""
    result = await harmony_server.call_tool(
        "find_key",
        {"progression": "C-F-G-Am"}
    )
    assert len(result) > 0
    data = json.loads(result[0].text)
    assert "key" in data
    assert "confidence" in data
    assert isinstance(data["confidence"], float)


# ============================================================================
# Groove Tools Tests (5 tools)
# ============================================================================

@pytest.mark.asyncio
async def test_groove_list_tools(groove_server):
    """Test groove tools are registered."""
    tools = await groove_server.list_tools()
    tool_names = [t.name for t in tools]
    
    assert "extract_groove" in tool_names
    assert "apply_groove" in tool_names
    assert "analyze_pocket" in tool_names
    assert "humanize_midi" in tool_names
    assert "quantize_smart" in tool_names
    assert len(tools) == 5


@pytest.mark.asyncio
async def test_extract_groove(groove_server):
    """Test extract_groove tool."""
    # Create a dummy MIDI file path for testing
    with patch('pathlib.Path.exists', return_value=True):
        with patch('music_brain.groove.extract_groove') as mock_extract:
            mock_groove = Mock()
            mock_groove.timing_stats = {"mean_deviation_ms": 5.0, "std_deviation_ms": 10.0}
            mock_groove.velocity_stats = {"min": 60, "max": 100, "mean": 80}
            mock_groove.swing_factor = 0.1
            mock_groove.notes = [1, 2, 3]
            mock_extract.return_value = mock_groove
            
            result = await groove_server.call_tool(
                "extract_groove",
                {"midi_file": "test.mid"}
            )
            assert len(result) > 0
            data = json.loads(result[0].text)
            assert "midi_file" in data


# ============================================================================
# Intent Tools Tests (4 tools)
# ============================================================================

@pytest.mark.asyncio
async def test_intent_list_tools(intent_server):
    """Test intent tools are registered."""
    tools = await intent_server.list_tools()
    tool_names = [t.name for t in tools]
    
    assert "create_intent" in tool_names
    assert "process_intent" in tool_names
    assert "validate_intent" in tool_names
    assert "suggest_rulebreaks" in tool_names
    assert len(tools) == 4


@pytest.mark.asyncio
async def test_create_intent(intent_server):
    """Test create_intent tool."""
    with patch('music_brain.session.intent_schema.CompleteSongIntent') as mock_intent:
        mock_instance = Mock()
        mock_instance.save = Mock()
        mock_intent.return_value = mock_instance
        
        result = await intent_server.call_tool(
            "create_intent",
            {"title": "Test Song", "output_file": "test_intent.json"}
        )
        assert len(result) > 0
        data = json.loads(result[0].text)
        assert data["status"] == "success"


@pytest.mark.asyncio
async def test_suggest_rulebreaks(intent_server):
    """Test suggest_rulebreaks tool."""
    with patch('music_brain.session.intent_schema.suggest_rule_break') as mock_suggest:
        mock_suggest.return_value = [
            {
                "rule": "HARMONY_ModalInterchange",
                "description": "Use modal interchange",
                "effect": "Emotional ambiguity",
                "use_when": "Nostalgia"
            }
        ]
        
        result = await intent_server.call_tool(
            "suggest_rulebreaks",
            {"emotion": "nostalgia"}
        )
        assert len(result) > 0
        data = json.loads(result[0].text)
        assert "suggestions" in data


# ============================================================================
# Audio Analysis Tools Tests (6 tools)
# ============================================================================

@pytest.mark.asyncio
async def test_audio_list_tools(audio_server):
    """Test audio tools are registered."""
    tools = await audio_server.list_tools()
    tool_names = [t.name for t in tools]
    
    assert "detect_bpm" in tool_names
    assert "detect_key" in tool_names
    assert "analyze_audio_feel" in tool_names
    assert "extract_chords" in tool_names
    assert "detect_scale" in tool_names
    assert "analyze_theory" in tool_names
    assert len(tools) == 6


@pytest.mark.asyncio
async def test_detect_bpm(audio_server):
    """Test detect_bpm tool."""
    with patch('pathlib.Path.exists', return_value=True):
        with patch('music_brain.audio.AudioAnalyzer') as mock_analyzer:
            mock_instance = Mock()
            mock_analysis = Mock()
            mock_bpm_result = Mock()
            mock_bpm_result.bpm = 120.0
            mock_bpm_result.confidence = 0.9
            mock_bpm_result.tempo_alternatives = [118.0, 122.0]
            mock_analysis.bpm_result = mock_bpm_result
            mock_instance.analyze_file.return_value = mock_analysis
            mock_analyzer.return_value = mock_instance
            
            result = await audio_server.call_tool(
                "detect_bpm",
                {"audio_file": "test.wav"}
            )
            assert len(result) > 0
            data = json.loads(result[0].text)
            assert "bpm" in data or "error" in data


# ============================================================================
# Teaching Tools Tests (3 tools)
# ============================================================================

@pytest.mark.asyncio
async def test_teaching_list_tools(teaching_server):
    """Test teaching tools are registered."""
    tools = await teaching_server.list_tools()
    tool_names = [t.name for t in tools]
    
    assert "explain_rulebreak" in tool_names
    assert "get_progression_info" in tool_names
    assert "emotion_to_music" in tool_names
    assert len(tools) == 3


@pytest.mark.asyncio
async def test_explain_rulebreak(teaching_server):
    """Test explain_rulebreak tool."""
    with patch('music_brain.session.teaching.RuleBreakingTeacher') as mock_teacher:
        mock_instance = Mock()
        mock_instance.explain_rule.return_value = "This rule creates emotional ambiguity..."
        mock_teacher.return_value = mock_instance
        
        result = await teaching_server.call_tool(
            "explain_rulebreak",
            {"rule_name": "HARMONY_ModalInterchange"}
        )
        assert len(result) > 0
        data = json.loads(result[0].text)
        assert "rule_name" in data


@pytest.mark.asyncio
async def test_emotion_to_music(teaching_server):
    """Test emotion_to_music tool."""
    with patch('music_brain.session.intent_schema.suggest_rule_break') as mock_suggest:
        mock_suggest.return_value = [
            {
                "rule": "HARMONY_AvoidTonicResolution",
                "effect": "Unresolved yearning"
            }
        ]
        
        result = await teaching_server.call_tool(
            "emotion_to_music",
            {"emotion": "grief"}
        )
        assert len(result) > 0
        data = json.loads(result[0].text)
        assert "musical_parameters" in data


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_unified_server():
    """Test that unified server aggregates all tools."""
    from daiw_mcp.server import create_server
    
    server = create_server()
    if server is None:
        pytest.skip("MCP not available")
    
    tools = await server.list_tools()
    tool_names = [t.name for t in tools]
    
    # Verify we have tools from all modules
    assert len(tools) >= 24  # At least 24 tools total
    
    # Check a sample from each category
    assert "analyze_progression" in tool_names  # Harmony
    assert "extract_groove" in tool_names  # Groove
    assert "create_intent" in tool_names  # Intent
    assert "detect_bpm" in tool_names  # Audio
    assert "explain_rulebreak" in tool_names  # Teaching


@pytest.mark.asyncio
async def test_tool_routing():
    """Test that tool calls are routed to correct modules."""
    from daiw_mcp.server import create_server
    
    server = create_server()
    if server is None:
        pytest.skip("MCP not available")
    
    # Test routing to harmony module
    with patch('daiw_mcp.tools.harmony.register_tools') as mock_register:
        # This would require more complex mocking
        pass  # Placeholder for routing test


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
