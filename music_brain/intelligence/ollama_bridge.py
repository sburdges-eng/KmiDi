"""
Ollama Bridge - Local NLM integration for lyrics, intent, and music generation.

Provides a simple interface to use local LLMs via Ollama for:
- Lyrics generation from emotional intent
- Intent parsing from natural language
- Creative suggestions and variations

Usage:
    from music_brain.intelligence.ollama_bridge import OllamaBridge
    
    bridge = OllamaBridge()
    lyrics = bridge.generate_lyrics(emotion="grief", theme="loss")
"""

import json
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import os


@dataclass
class OllamaConfig:
    """Configuration for Ollama connection."""
    
    model: str = "mistral"  # Default model (Apache 2.0 license)
    host: str = "http://localhost:11434"
    timeout_seconds: int = 60
    temperature: float = 0.7
    max_tokens: int = 512
    
    @classmethod
    def from_env(cls) -> "OllamaConfig":
        """Load configuration from environment variables."""
        return cls(
            model=os.getenv("OLLAMA_MODEL", "mistral"),
            host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            timeout_seconds=int(os.getenv("OLLAMA_TIMEOUT", "60")),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("OLLAMA_MAX_TOKENS", "512")),
        )


class OllamaBridge:
    """
    Bridge to Ollama for local NLM inference.
    
    Requires Ollama to be installed and running:
        brew install ollama
        ollama pull mistral
        ollama serve
    """
    
    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig.from_env()
        self._available = None
    
    def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        if self._available is not None:
            return self._available
        
        try:
            import requests
            response = requests.get(
                f"{self.config.host}/api/tags",
                timeout=5
            )
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "").split(":")[0] for m in models]
                self._available = self.config.model in model_names
            else:
                self._available = False
        except Exception:
            self._available = False
        
        return self._available
    
    def _call_ollama(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Optional[str]:
        """Make a call to Ollama API."""
        try:
            import requests
        except ImportError:
            return None
        
        if not self.is_available():
            return None
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = requests.post(
                f"{self.config.host}/api/chat",
                json={
                    "model": self.config.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature or self.config.temperature,
                        "num_predict": max_tokens or self.config.max_tokens,
                    }
                },
                timeout=self.config.timeout_seconds,
            )
            
            if response.status_code == 200:
                return response.json().get("message", {}).get("content", "")
            return None
        except Exception:
            return None
    
    def generate_lyrics(
        self,
        emotion: str,
        theme: Optional[str] = None,
        style: Optional[str] = None,
        lines: int = 4,
    ) -> Optional[str]:
        """
        Generate lyrics based on emotional intent.
        
        Args:
            emotion: Primary emotion (grief, joy, anger, etc.)
            theme: Optional theme (loss, love, freedom, etc.)
            style: Optional style (folk, hip-hop, pop, etc.)
            lines: Number of lines to generate
        
        Returns:
            Generated lyrics or None if unavailable
        """
        system_prompt = """You are a skilled songwriter. Generate lyrics that:
- Capture the specified emotion authentically
- Use concrete imagery over abstract concepts
- Follow natural speech rhythms
- Avoid clichés and generic phrases
Only output the lyrics, no explanations or titles."""

        prompt_parts = [f"Write {lines} lines of lyrics expressing {emotion}."]
        if theme:
            prompt_parts.append(f"Theme: {theme}.")
        if style:
            prompt_parts.append(f"Style: {style}.")
        
        return self._call_ollama(
            prompt=" ".join(prompt_parts),
            system_prompt=system_prompt,
            temperature=0.8,  # Slightly higher for creativity
        )
    
    def parse_intent(
        self,
        user_input: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Parse natural language into structured song intent.
        
        Args:
            user_input: Free-form description of what the user wants
        
        Returns:
            Structured intent dictionary or None if unavailable
        """
        system_prompt = """You are a music production assistant. Parse the user's input into structured song intent.
Return ONLY valid JSON with these fields:
{
    "mood_primary": "main emotion",
    "mood_secondary": "secondary emotion or null",
    "genre": "music genre",
    "tempo_feel": "slow/medium/fast",
    "key_suggestion": "major/minor",
    "themes": ["theme1", "theme2"],
    "rule_breaks": ["suggested rule to break for emotional effect"]
}"""

        result = self._call_ollama(
            prompt=f"Parse this song idea: {user_input}",
            system_prompt=system_prompt,
            temperature=0.3,  # Lower for structured output
        )
        
        if not result:
            return None
        
        # Try to extract JSON from response
        try:
            # Handle case where model wraps in markdown
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0]
            elif "```" in result:
                result = result.split("```")[1].split("```")[0]
            return json.loads(result.strip())
        except json.JSONDecodeError:
            return None
    
    def suggest_chord_progression(
        self,
        emotion: str,
        genre: Optional[str] = None,
        context: Optional[str] = None,
    ) -> Optional[List[str]]:
        """
        Suggest chord progressions for an emotion.
        
        Args:
            emotion: Target emotion
            genre: Optional genre constraint
            context: Optional context (verse, chorus, bridge)
        
        Returns:
            List of chord suggestions or None
        """
        system_prompt = """You are a music theory expert. Suggest chord progressions.
Return ONLY a JSON array of roman numerals, e.g.: ["I", "V", "vi", "IV"]
Include borrowed chords or modal interchange if appropriate for the emotion."""

        prompt_parts = [f"Suggest a chord progression for {emotion}."]
        if genre:
            prompt_parts.append(f"Genre: {genre}.")
        if context:
            prompt_parts.append(f"Section: {context}.")
        
        result = self._call_ollama(
            prompt=" ".join(prompt_parts),
            system_prompt=system_prompt,
            temperature=0.5,
        )
        
        if not result:
            return None
        
        try:
            # Handle markdown wrapping
            if "```" in result:
                result = result.split("```")[1].split("```")[0]
                if result.startswith("json"):
                    result = result[4:]
            return json.loads(result.strip())
        except json.JSONDecodeError:
            return None
    
    def explain_rule_break(
        self,
        rule: str,
        emotion: str,
    ) -> Optional[str]:
        """
        Explain why a rule should be broken for emotional effect.
        
        Args:
            rule: The rule being broken (e.g., "HARMONY_AvoidTonicResolution")
            emotion: The target emotion
        
        Returns:
            Explanation string or None
        """
        system_prompt = """You are a music theory teacher explaining intentional rule-breaking.
Be specific about the emotional effect and give concrete examples."""

        prompt = f"Explain why breaking '{rule}' is effective for expressing {emotion}."
        
        return self._call_ollama(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.6,
        )
    
    def generate_melody_suggestions(
        self,
        emotion: str,
        key: str,
        mode: str,
        phrase_length: int = 4,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate melodic suggestions based on emotional intent.
        
        Args:
            emotion: Target emotion
            key: Musical key (C, F#, etc.)
            mode: major or minor
            phrase_length: Bars in the phrase
        
        Returns:
            Dictionary with melodic suggestions or None
        """
        system_prompt = """You are a melody composition assistant.
Suggest melodic characteristics that match the emotion.
Return ONLY valid JSON with these fields:
{
    "contour": "ascending/descending/arch/wave",
    "range_notes": 8,
    "leap_frequency": "rare/moderate/frequent",
    "rhythmic_density": "sparse/moderate/dense",
    "resolution_type": "strong/weak/suspended",
    "suggested_intervals": ["m3", "P5", "M6"],
    "avoid_intervals": ["tritone", "M7"],
    "guidance": "Brief description of melodic approach"
}"""

        prompt = f"Melodic suggestions for {emotion} in {key} {mode}, {phrase_length} bars."
        
        result = self._call_ollama(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.4,
        )
        
        if not result:
            return None
        
        try:
            if "```" in result:
                result = result.split("```")[1].split("```")[0]
                if result.startswith("json"):
                    result = result[4:]
            return json.loads(result.strip())
        except json.JSONDecodeError:
            return None


def create_ollama_bridge(config: Optional[OllamaConfig] = None) -> OllamaBridge:
    """Factory function to create OllamaBridge."""
    return OllamaBridge(config)


# Convenience functions for quick access
def generate_lyrics(emotion: str, **kwargs) -> Optional[str]:
    """Quick lyrics generation."""
    bridge = OllamaBridge()
    return bridge.generate_lyrics(emotion, **kwargs)


def parse_intent(user_input: str) -> Optional[Dict[str, Any]]:
    """Quick intent parsing."""
    bridge = OllamaBridge()
    return bridge.parse_intent(user_input)
