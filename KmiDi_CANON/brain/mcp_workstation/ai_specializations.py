"""
AI specializations for intelligent task assignment in the orchestrator.

This module provides agent capability definitions and task routing logic.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple

from .models import AIAgent


class TaskType(Enum):
    """Types of tasks that can be assigned to agents."""
    LLM = "llm"
    MIDI = "midi"
    IMAGE = "image"
    AUDIO = "audio"


@dataclass
class AgentCapabilities:
    """
    Detailed capabilities and characteristics of an AI agent.

    Used for intelligent task routing based on agent specializations.
    """
    display_name: str = ""
    description: str = ""
    strengths: Dict[str, str] = field(default_factory=dict)
    proposal_categories: List[str] = field(default_factory=list)
    best_languages: List[str] = field(default_factory=list)
    special_abilities: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    recommended_for: List[str] = field(default_factory=list)


# Agent capability definitions
_AGENT_CAPABILITIES = {
    AIAgent.LLM: AgentCapabilities(
        display_name="LLM Reasoning Engine",
        description="Processes natural language into structured musical intent",
        strengths={
            "text_parsing": "Excellent at understanding user text and emotional context",
            "intent_extraction": "Maps mood, tension, and narrative to musical parameters",
            "prompt_generation": "Creates detailed prompts for image/audio generation",
        },
        proposal_categories=["text_analysis", "intent_parsing", "prompt_creation"],
        best_languages=["english", "natural_language"],
        special_abilities=[
            "Emotional sentiment analysis",
            "Rule-breaking justification extraction",
            "Image/audio prompt generation",
            "Nested intent structure creation",
        ],
        limitations=[
            "Requires model file or llama.cpp backend",
            "May use rule-based fallback if model unavailable",
            "Limited by training data quality",
        ],
        recommended_for=[
            "User text → CompleteSongIntent conversion",
            "Generating descriptive prompts for multimodal generation",
            "Analyzing lyrical content for emotional context",
        ]
    ),

    AIAgent.MIDI: AgentCapabilities(
        display_name="MIDI Generation Pipeline",
        description="Transforms musical intent into MIDI files",
        strengths={
            "harmony_generation": "Creates chord progressions with intentional rule-breaking",
            "groove_processing": "Generates rhythmic patterns with timing/velocity",
            "structure_design": "Builds arrangement sections and dynamics",
        },
        proposal_categories=["music_theory", "midi_generation", "harmony", "rhythm"],
        best_languages=["music_theory", "chord_notation"],
        special_abilities=[
            "Modal interchange and borrowed chords",
            "Polyrhythmic layer generation",
            "Extreme dynamic range handling",
            "Melodic contour creation",
        ],
        limitations=[
            "Requires valid CompleteSongIntent input",
            "Groove humanization not yet fully implemented",
            "Limited to MIDI format (no audio rendering)",
        ],
        recommended_for=[
            "CompleteSongIntent → MIDI file conversion",
            "Harmony and rhythm generation",
            "Structural arrangement design",
        ]
    ),

    AIAgent.IMAGE: AgentCapabilities(
        display_name="Image Generation Engine",
        description="Creates visual representations of musical intent",
        strengths={
            "prompt_understanding": "Translates musical mood to visual concepts",
            "style_matching": "Aligns imagery with emotional tone",
            "composition": "Creates balanced and evocative compositions",
        },
        proposal_categories=["image_generation", "visual_design", "mood_visualization"],
        best_languages=["diffusion_prompts", "visual_descriptors"],
        special_abilities=[
            "Stable Diffusion integration",
            "Mood-driven prompt enhancement",
            "Multiple aspect ratio support",
        ],
        limitations=[
            "Requires diffusers library and model checkpoint",
            "Stubbed if dependencies not installed",
            "GPU recommended for reasonable speed",
        ],
        recommended_for=[
            "Album art generation",
            "Mood visualization",
            "Emotional imagery creation from intent",
        ]
    ),

    AIAgent.AUDIO: AgentCapabilities(
        display_name="Audio Generation Engine",
        description="Generates audio textures and atmospheres",
        strengths={
            "texture_creation": "Generates ambient soundscapes and atmospheres",
            "prompt_interpretation": "Converts textural descriptions to audio",
            "style_transfer": "Applies audio characteristics from prompts",
        },
        proposal_categories=["audio_generation", "texture", "atmosphere"],
        best_languages=["audio_descriptors", "texture_language"],
        special_abilities=[
            "MusicGen integration (when available)",
            "Texture-driven generation",
            "Atmosphere and mood audio",
        ],
        limitations=[
            "Requires audiocraft library",
            "Stubbed if audiocraft not installed",
            "Slower generation on CPU",
        ],
        recommended_for=[
            "Atmospheric intros and outros",
            "Textural bed tracks",
            "Ambient soundscape generation",
        ]
    ),
}


def get_capabilities(agent: AIAgent) -> AgentCapabilities:
    """
    Get the detailed capabilities of a specific agent.

    Args:
        agent: The agent to get capabilities for

    Returns:
        AgentCapabilities object with detailed information
    """
    return _AGENT_CAPABILITIES.get(agent, AgentCapabilities(
        display_name=agent.value,
        description="Unknown agent type"
    ))


def suggest_task_assignment(tasks: List[Tuple[str, TaskType]]) -> Dict[str, AIAgent]:
    """
    Intelligently assign tasks to appropriate agents based on task type and agent capabilities.

    This function implements a simple but effective task routing strategy:
    1. Direct type matching (MIDI tasks → MIDI agent)
    2. Capability-based routing for ambiguous tasks
    3. Load balancing across agents when multiple can handle a task

    Args:
        tasks: List of (task_id, task_type) tuples

    Returns:
        Dict mapping task_id → assigned AIAgent

    Example:
        >>> tasks = [
        ...     ("parse_user_text", TaskType.LLM),
        ...     ("generate_midi", TaskType.MIDI),
        ...     ("create_cover_art", TaskType.IMAGE),
        ... ]
        >>> assignments = suggest_task_assignment(tasks)
        >>> assignments["parse_user_text"]
        <AIAgent.LLM: 'llm'>
    """
    if not tasks:
        return {}

    # Direct task type → agent type mapping
    type_to_agent = {
        TaskType.LLM: AIAgent.LLM,
        TaskType.MIDI: AIAgent.MIDI,
        TaskType.IMAGE: AIAgent.IMAGE,
        TaskType.AUDIO: AIAgent.AUDIO,
    }

    assignments = {}
    agent_load = {agent: 0 for agent in AIAgent}  # Track load for balancing

    for task_id, task_type in tasks:
        # Primary routing: direct type match
        if task_type in type_to_agent:
            assigned_agent = type_to_agent[task_type]
            assignments[task_id] = assigned_agent
            agent_load[assigned_agent] += 1
        else:
            # Fallback: assign to least loaded capable agent
            # For unknown task types, default to LLM (most versatile)
            assigned_agent = min(agent_load.items(), key=lambda x: x[1])[0]
            assignments[task_id] = assigned_agent
            agent_load[assigned_agent] += 1

    return assignments


def get_agent_for_task_type(task_type: TaskType) -> AIAgent:
    """
    Get the recommended agent for a specific task type.

    Args:
        task_type: Type of task

    Returns:
        The recommended AIAgent for this task type
    """
    mapping = {
        TaskType.LLM: AIAgent.LLM,
        TaskType.MIDI: AIAgent.MIDI,
        TaskType.IMAGE: AIAgent.IMAGE,
        TaskType.AUDIO: AIAgent.AUDIO,
    }
    return mapping.get(task_type, AIAgent.LLM)  # Default to LLM if unknown


def get_compatible_agents(task_type: TaskType) -> List[AIAgent]:
    """
    Get all agents that can handle a specific task type.

    For most tasks, only one agent is compatible. This allows for future
    expansion where multiple agents might share capabilities.

    Args:
        task_type: Type of task

    Returns:
        List of compatible agents (primary agent first)
    """
    primary = get_agent_for_task_type(task_type)
    return [primary]  # For now, one-to-one mapping


def get_task_priority_score(task_type: TaskType, agent: AIAgent) -> float:
    """
    Calculate a priority score for assigning a task type to an agent.

    Higher scores indicate better fit. Used for load balancing when
    multiple agents can handle a task.

    Args:
        task_type: Type of task
        agent: Agent to score

    Returns:
        Priority score (0.0-1.0), where 1.0 is perfect match
    """
    # Direct type match = highest priority
    optimal_agent = get_agent_for_task_type(task_type)
    if agent == optimal_agent:
        return 1.0

    # No cross-compatibility in current design
    return 0.0


def format_capability_report(agent: AIAgent) -> str:
    """
    Generate a human-readable capability report for an agent.

    Args:
        agent: Agent to generate report for

    Returns:
        Formatted capability report string
    """
    caps = get_capabilities(agent)

    report_lines = [
        f"Agent: {caps.display_name}",
        f"Description: {caps.description}",
        "",
        "Strengths:",
    ]

    for strength, desc in caps.strengths.items():
        report_lines.append(f"  - {strength}: {desc}")

    if caps.special_abilities:
        report_lines.append("")
        report_lines.append("Special Abilities:")
        for ability in caps.special_abilities:
            report_lines.append(f"  - {ability}")

    if caps.limitations:
        report_lines.append("")
        report_lines.append("Limitations:")
        for limitation in caps.limitations:
            report_lines.append(f"  - {limitation}")

    if caps.recommended_for:
        report_lines.append("")
        report_lines.append("Recommended For:")
        for rec in caps.recommended_for:
            report_lines.append(f"  - {rec}")

    return "\n".join(report_lines)
