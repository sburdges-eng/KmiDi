"""
Cognitive Router — HOW MUCH INTELLIGENCE IS REQUIRED?

Routes requests to REFLEX (deterministic), SKILL (bounded capability), or DEEP (expensive reasoning).
No tools. No agents. Just routing. Protects real-time paths from LLM/cognition.

Rule: Never let deep reasoning or tool selection into audio threads or low-latency pipelines.
See: .cursor/rules/tool-minimization.mdc
"""

from enum import Enum


class CognitiveMode(Enum):
    """Reflex = 0 intelligence, deterministic. Skill = bounded. Deep = expensive cognition."""
    REFLEX = "reflex"
    SKILL = "skill"
    DEEP = "deep_reasoning"


class CognitiveRouter:
    """
    Routes incoming requests to the appropriate cognitive mode.
    Start with rules; complexity score can be trained later.
    """

    def __init__(
        self,
        reflex_ambiguity_max: float = 0.2,
        reflex_creativity_max: float = 0.2,
        skill_creativity_max: float = 0.6,
    ):
        self.reflex_ambiguity_max = reflex_ambiguity_max
        self.reflex_creativity_max = reflex_creativity_max
        self.skill_creativity_max = skill_creativity_max

    def route(self, request: dict) -> CognitiveMode:
        """Classify request → REFLEX | SKILL | DEEP. No tools, no orchestration."""
        if self._is_reflex(request):
            return CognitiveMode.REFLEX
        if self._is_skill(request):
            return CognitiveMode.SKILL
        return CognitiveMode.DEEP

    def _is_reflex(self, r: dict) -> bool:
        """Structured, low ambiguity/creativity → reflex pipeline."""
        return (
            r.get("structured") is True
            and r.get("ambiguity", 0) < self.reflex_ambiguity_max
            and r.get("creativity", 0) < self.reflex_creativity_max
        )

    def _is_skill(self, r: dict) -> bool:
        """Known domain, bounded creativity → skill execution."""
        return (
            r.get("domain_known") is True
            and r.get("creativity", 0) < self.skill_creativity_max
        )

    def complexity_score(self, r: dict) -> float:
        """
        Optional: ambiguity + creativity + novelty + emotional_weight.
        If score > threshold → deep. Start dumb; can train later.
        """
        return (
            r.get("ambiguity", 0)
            + r.get("creativity", 0)
            + r.get("novelty", 0)
            + r.get("emotional_weight", 0)
        )
