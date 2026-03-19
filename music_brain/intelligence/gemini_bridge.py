"""
Gemini Bridge - Google Gemini integration for creative music generation.

Provides an interface to Gemini AI for:
- Intent enhancement and creative brainstorming
- Lyrics generation with emotional depth
- Harmonic and rhythmic variation suggestions

Usage:
    from music_brain.intelligence.gemini_bridge import GeminiBridge

    bridge = GeminiBridge()
    lyrics = await bridge.generate_lyrics(emotion="grief", theme="loss")
"""

import os
import json
import logging
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class GeminiConfig:
    """Configuration for Gemini API connection."""
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    model: str = "gemini-1.5-pro"  # Default to latest high-capability model
    temperature: float = 0.7
    max_output_tokens: int = 1024
    top_p: float = 0.95
    top_k: int = 40

    @classmethod
    def from_env(cls) -> "GeminiConfig":
        """Load configuration from environment variables."""
        return cls(
            api_key=os.getenv("GEMINI_API_KEY"),
            model=os.getenv("GEMINI_MODEL", "gemini-1.5-pro"),
            temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.7")),
            max_output_tokens=int(os.getenv("GEMINI_MAX_TOKENS", "1024")),
        )


class GeminiBridge:
    """
    Bridge to Google Gemini for AI-powered musical creativity.

    Integrates with the official `google-generativeai` package if available,
    otherwise can fall back to direct HTTP requests or mock responses.
    """

    def __init__(self, config: Optional[GeminiConfig] = None):
        self.config = config or GeminiConfig.from_env()
        self._logger = logging.getLogger("music_brain.intelligence.gemini")
        self._model = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize the Gemini client."""
        if not self.config.api_key:
            self._logger.warning("GEMINI_API_KEY not found in environment")
            return False

        try:
            import google.generativeai as genai
            genai.configure(api_key=self.config.api_key)
            self._model = genai.GenerativeModel(self.config.model)
            self._initialized = True
            return True
        except ImportError:
            self._logger.error("google-generativeai package not installed")
            return False
        except Exception as e:
            self._logger.error(f"Failed to initialize Gemini: {e}")
            return False

    async def generate_content(
        self,
        prompt: str,
        system_instruction: Optional[str] = None
    ) -> Optional[str]:
        """Generic content generation with Gemini."""
        if not self._initialized:
            if not await self.initialize():
                return self._mock_response(prompt)

        try:
            # Note: For real implementation, use the async version of the SDK
            # if available, or run the synchronous call in a thread pool.
            import google.generativeai as genai
            
            # Setup generation config
            gen_config = {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "max_output_tokens": self.config.max_output_tokens,
            }

            # If system_instruction is provided, we use the newer SDK pattern
            if system_instruction:
                model = genai.GenerativeModel(
                    model_name=self.config.model,
                    system_instruction=system_instruction
                )
            else:
                model = self._model

            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config=gen_config
            )
            
            return response.text
        except Exception as e:
            self._logger.error(f"Gemini generation failed: {e}")
            return None

    async def generate_lyrics(self, emotion: str, theme: str) -> Optional[str]:
        """Generate lyrics based on emotional intent."""
        system = "You are a poetic songwriter specializing in emotional depth and musical phrasing."
        prompt = f"Write song lyrics for a composition about {theme}. The primary emotion is {emotion}."
        return await self.generate_content(prompt, system_instruction=system)

    async def enhance_intent(self, raw_intent: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance a raw musical intent with creative suggestions."""
        system = "You are a music production assistant specializing in translating emotions into technical parameters."
        prompt = f"Given this raw musical intent: {json.dumps(raw_intent)}, provide suggestions for key, mode, instrumentation, and emotional textures."
        
        response = await self.generate_content(prompt, system_instruction=system)
        if response:
            # In a real implementation, we'd prompt for JSON and parse it
            return {"enhanced_suggestion": response, "original": raw_intent}
        return raw_intent

    def _mock_response(self, prompt: str) -> str:
        """Provide a mock response when Gemini is unavailable (for testing)."""
        return f"[MOCK GEMINI RESPONSE] Re: {prompt[:50]}... (GEMINI_API_KEY missing)"


async def main():
    """Simple test script for GeminiBridge."""
    logging.basicConfig(level=logging.INFO)
    bridge = GeminiBridge()
    print("Initializing Gemini Bridge...")
    
    # Test lyrics generation
    lyrics = await bridge.generate_lyrics("grief", "ocean waves")
    print(f"\nGenerated Lyrics:\n{lyrics}")


if __name__ == "__main__":
    asyncio.run(main())
