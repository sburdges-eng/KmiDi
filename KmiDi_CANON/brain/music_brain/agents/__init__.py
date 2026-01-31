"""
DAiW Agents Module

AI-powered music production agents and integration bridges.

Components:
- daiw_mcp_server: MCP server for voice synthesis tools
- ableton_bridge: Ableton Live OSC/MIDI integration
- crewai_music_agents: Multi-agent music production crew
- unified_hub: Central orchestration hub

Quick Start:
    from music_brain.agents import UnifiedHub, start_hub, stop_hub

    # Start the hub
    hub = start_hub()

    # Voice synthesis
    hub.speak("Hello world")
    hub.note_on(60, 100)

    # DAW control
    hub.connect_daw()
    hub.daw_play()

    # Agent tasks
    await hub.run_agent_task("produce a vocal track", task_type="voice")

    # IMPORTANT: Always stop when done
    stop_hub()

Shutdown Methods:
    # Stop just the hub
    stop_hub()

    # Force immediate stop
    force_stop_hub()

    # Complete shutdown of all systems
    shutdown_all()

Context Manager Usage:
    # Automatic cleanup with context manager
    with UnifiedHub() as hub:
        hub.speak("Hello")
        hub.connect_daw()
        # ... use the hub ...
    # Automatically stopped when exiting context

    # Same for Ableton bridge
    with DAiWAbletonIntegration() as bridge:
        bridge.osc_bridge.play()
        # ...
    # Automatically disconnected
"""

# Core hub
from .unified_hub import (
    UnifiedHub,
    SessionConfig,
    VoiceState,
    DAWState,
    HubState,
    AudioRoutingMode,
    get_hub,
    start_hub,
    stop_hub,
    force_stop_hub,
    shutdown_all
)

# Ableton integration
from .ableton_bridge import (
    AbletonOSCBridge,
    AbletonMIDIBridge,
    DAiWAbletonIntegration,
    AbletonTrackInfo,
    AbletonClipInfo,
    AbletonDeviceInfo,
    AbletonConnectionState,
    check_ableton_connection,
    get_default_cc_mappings,
    create_ableton_mcp_tools
)

# CrewAI agents
from .crewai_music_agents import (
    AgentRole,
    AgentConfig,
    VoiceSynthesisTool,
    AbletonControlTool,
    CompositionTool,
    MixingTool,
    LyricsTool,
    ToolManager,
    create_agent,
    create_music_production_crew,
    create_crewai_tools,
    create_voice_production_task,
    create_song_production_task,
    create_vocal_direction_task,
    run_production_task,
    get_available_roles,
    get_role_info,
    check_dependencies,
    get_tool_manager,
    shutdown_tools
)

__all__ = [
    # Hub
    "UnifiedHub",
    "SessionConfig",
    "VoiceState",
    "DAWState",
    "HubState",
    "AudioRoutingMode",
    "get_hub",
    "start_hub",
    "stop_hub",
    "force_stop_hub",
    "shutdown_all",

    # Ableton
    "AbletonOSCBridge",
    "AbletonMIDIBridge",
    "DAiWAbletonIntegration",
    "AbletonTrackInfo",
    "AbletonClipInfo",
    "AbletonDeviceInfo",
    "AbletonConnectionState",
    "check_ableton_connection",
    "get_default_cc_mappings",
    "create_ableton_mcp_tools",

    # CrewAI
    "AgentRole",
    "AgentConfig",
    "VoiceSynthesisTool",
    "AbletonControlTool",
    "CompositionTool",
    "MixingTool",
    "LyricsTool",
    "ToolManager",
    "create_agent",
    "create_music_production_crew",
    "create_crewai_tools",
    "create_voice_production_task",
    "create_song_production_task",
    "create_vocal_direction_task",
    "run_production_task",
    "get_available_roles",
    "get_role_info",
    "check_dependencies",
    "get_tool_manager",
    "shutdown_tools",
]
