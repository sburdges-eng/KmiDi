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

import hashlib
import os
import json
import logging
import asyncio
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple


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

    # Rate limiting: token-bucket (requests per minute)
    RATE_LIMIT_RPM: int = 10
    # Response cache: LRU with TTL
    CACHE_MAX_SIZE: int = 64
    CACHE_TTL_SECONDS: float = 300.0  # 5 minutes

    def __init__(self, config: Optional[GeminiConfig] = None):
        self.config = config or GeminiConfig.from_env()
        self._logger = logging.getLogger("music_brain.intelligence.gemini")
        self._model = None
        self._initialized = False
        # Rate limiting state
        self._request_timestamps: List[float] = []
        # Response cache: key -> (response, timestamp)
        self._response_cache: OrderedDict[str, Tuple[str, float]] = OrderedDict()

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

    def _cache_key(self, prompt: str, system_instruction: Optional[str]) -> str:
        """Compute a stable cache key from prompt + system instruction."""
        raw = f"{system_instruction or ''}||{prompt}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def _cache_get(self, key: str) -> Optional[str]:
        """Return cached response if still valid, or None."""
        entry = self._response_cache.get(key)
        if entry is None:
            return None
        response, ts = entry
        if (time.monotonic() - ts) > self.CACHE_TTL_SECONDS:
            del self._response_cache[key]
            return None
        # Move to end (most recently used)
        self._response_cache.move_to_end(key)
        return response

    def _cache_put(self, key: str, response: str) -> None:
        """Store a response in the LRU cache."""
        self._response_cache[key] = (response, time.monotonic())
        while len(self._response_cache) > self.CACHE_MAX_SIZE:
            self._response_cache.popitem(last=False)

    def _check_rate_limit(self) -> bool:
        """Return True if within rate limit, False if throttled."""
        now = time.monotonic()
        # Remove timestamps older than 60 seconds
        self._request_timestamps = [t for t in self._request_timestamps if (now - t) < 60.0]
        if len(self._request_timestamps) >= self.RATE_LIMIT_RPM:
            return False
        self._request_timestamps.append(now)
        return True

    async def generate_content(
        self,
        prompt: str,
        system_instruction: Optional[str] = None
    ) -> Optional[str]:
        """Generic content generation with Gemini (cached, rate-limited)."""
        # Check cache first
        cache_key = self._cache_key(prompt, system_instruction)
        cached = self._cache_get(cache_key)
        if cached is not None:
            self._logger.debug("Gemini cache hit for key %s", cache_key[:8])
            return cached

        if not self._initialized:
            if not await self.initialize():
                return self._mock_response(prompt)

        # Rate limiting
        if not self._check_rate_limit():
            self._logger.warning("Gemini rate limit reached (%d RPM)", self.RATE_LIMIT_RPM)
            return None

        try:
            import google.generativeai as genai

            gen_config = {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "max_output_tokens": self.config.max_output_tokens,
            }

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

            text = response.text
            # Cache the response
            if text:
                self._cache_put(cache_key, text)
            return text
        except Exception as e:
            self._logger.error("Gemini generation failed: %s", e)
            return None

    async def generate_lyrics(self, emotion: str, theme: str) -> Optional[str]:
        """Generate lyrics based on emotional intent."""
        system = "You are a poetic songwriter specializing in emotional depth and musical phrasing."
        prompt = f"Write song lyrics for a composition about {theme}. The primary emotion is {emotion}."
        return await self.generate_content(prompt, system_instruction=system)

    async def enhance_intent(self, raw_intent: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance a raw musical intent with creative suggestions."""
        system = "You are a music production assistant specializing in translating emotions into technical parameters. Return ONLY valid JSON."
        prompt = f"Given this raw musical intent: {json.dumps(raw_intent)}, provide suggestions for key, mode, instrumentation, and emotional textures."

        response = await self.generate_content(prompt, system_instruction=system)
        if response:
            parsed = self._extract_json(response)
            if parsed is not None:
                parsed["original"] = raw_intent
                return parsed
            return {"enhanced_suggestion": response, "original": raw_intent}
        return raw_intent

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        """Robust JSON extraction: try direct parse, regex, then markdown fence."""
        # 1. Direct parse
        try:
            return json.loads(text.strip())
        except (json.JSONDecodeError, ValueError):
            pass
        # 2. Regex: find first {...} block
        match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except (json.JSONDecodeError, ValueError):
                pass
        # 3. Markdown fence fallback
        if "```" in text:
            parts = text.split("```")
            for part in parts[1::2]:
                cleaned = part.strip()
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
                try:
                    return json.loads(cleaned)
                except (json.JSONDecodeError, ValueError):
                    continue
        return None

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
