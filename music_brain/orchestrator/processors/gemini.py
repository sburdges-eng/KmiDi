"""
Gemini Processor for AI Orchestrator.

Uses Google Gemini to enhance musical intent and provide creative suggestions.

Usage:
    from music_brain.orchestrator.processors import GeminiProcessor

    processor = GeminiProcessor()
    result = await processor.process(intent_data, context)
"""

from typing import Any, Dict, Optional
from dataclasses import dataclass

from music_brain.orchestrator.processors.base import BaseProcessor
from music_brain.orchestrator.interfaces import (
    ProcessorConfig,
    ProcessorResult,
    ExecutionContext,
)
from music_brain.intelligence.gemini_bridge import GeminiBridge


class GeminiProcessor(BaseProcessor):
    """
    Processor that uses Gemini to creatively enhance musical parameters.

    This processor can be inserted into an AIOrchestrator pipeline to:
    - Generate poetic lyrics based on current context
    - Suggest rule-breaking creative directions
    - Translate emotional descriptors into technical MIDI/Audio parameters
    """

    def __init__(self, config: Optional[ProcessorConfig] = None):
        config = config or ProcessorConfig(name="gemini")
        super().__init__(config)
        self.bridge = GeminiBridge()

    async def _validate_impl(self, input_data: Any) -> bool:
        """Validate input - needs at least some text or emotion."""
        if isinstance(input_data, dict):
            return bool(input_data.get("prompt") or input_data.get("emotion"))
        return isinstance(input_data, str)

    async def _process_impl(
        self,
        input_data: Any,
        context: ExecutionContext,
    ) -> ProcessorResult:
        """
        Process data using Gemini.

        Args:
            input_data: Prompt string or dict with 'prompt', 'emotion', 'theme'
            context: Execution context

        Returns:
            ProcessorResult with Gemini output
        """
        try:
            # Determine prompt
            if isinstance(input_data, str):
                prompt = input_data
                system = "You are a creative music production assistant."
            else:
                emotion = input_data.get("emotion", context.get_shared("emotion", "neutral"))
                theme = input_data.get("theme", "abstract")
                prompt = input_data.get("prompt", f"Enhance this musical direction: {emotion} theme: {theme}")
                system = input_data.get("system", "You are a creative music production assistant.")

            self._logger.info(f"Gemini processing prompt: {prompt[:50]}...")

            # Call Gemini
            response = await self.bridge.generate_content(prompt, system_instruction=system)

            if response:
                # Store in shared context
                context.set_shared("gemini_output", response)
                
                return ProcessorResult(
                    success=True,
                    data=response,
                    metadata={
                        "model": self.bridge.config.model,
                        "prompt_length": len(prompt),
                    },
                )
            else:
                return ProcessorResult(
                    success=False,
                    error="Gemini returned no content",
                )

        except Exception as e:
            self._logger.error(f"Gemini processor failed: {e}")
            return ProcessorResult(
                success=False,
                error=str(e),
            )
