"""
ONNX LLM integration for Music Brain.

Thin wrapper around onnxruntime-genai to run LLaMA-style ONNX models locally.
Designed for deployment in the dedicated LLM service container but also works
in-process when dependencies are available.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import os
import time


@dataclass
class OnnxLLMConfig:
    """Configuration for ONNX LLM runtime."""

    model_path: str
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    @classmethod
    def from_env(cls) -> "OnnxLLMConfig":
        """Load configuration from environment variables."""
        default_path = Path.home() / ".idaw" / "models" / "llama3-onnx"

        return cls(
            model_path=os.getenv("LLM_ONNX_MODEL_PATH", str(default_path)),
            max_length=int(os.getenv("LLM_ONNX_MAX_LENGTH", "512")),
            temperature=float(os.getenv("LLM_ONNX_TEMPERATURE", "0.7")),
            top_p=float(os.getenv("LLM_ONNX_TOP_P", "0.9")),
        )


class OnnxGenAILLM:
    """Simple text-generation wrapper using onnxruntime-genai."""

    def __init__(self, config: Optional[OnnxLLMConfig] = None):
        self.config = config or OnnxLLMConfig.from_env()
        self._model = None
        self._tokenizer = None
        self._available = False
        self._load()

    @property
    def is_available(self) -> bool:
        """Return True when model and tokenizer are ready."""
        return self._available

    def _load(self) -> None:
        """Load model and tokenizer if possible."""
        try:
            import onnxruntime_genai as og
        except ImportError:
            print("onnxruntime-genai not installed. Install to enable ONNX LLM.")
            return

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            print(f"ONNX model path not found: {model_path}")
            return

        try:
            self._model = og.Model(str(model_path))
            self._tokenizer = og.Tokenizer(self._model)
            self._available = True
        except Exception as exc:  # pragma: no cover - hardware/env specific
            print(f"Failed to load ONNX LLM: {exc}")
            self._available = False

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat-style messages to a single prompt string."""
        prompt_parts = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            prompt_parts.append(f"{role}: {content}")
        prompt_parts.append("assistant:")
        return "\n".join(prompt_parts)

    def generate(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Dict[str, object]:
        """
        Generate text for a single prompt.

        Returns dict with output text and latency_ms. Falls back with a message
        when the model is unavailable.
        """
        if not self._available:
            return {
                "output": "[ONNX LLM unavailable. Check model path and dependencies.]",
                "latency_ms": 0.0,
            }

        try:
            import onnxruntime_genai as og
        except ImportError:
            return {
                "output": "[onnxruntime-genai missing. Install to enable ONNX LLM.]",
                "latency_ms": 0.0,
            }

        params = og.GeneratorParams(self._model)
        params.set_search_options(
            max_length=max_length or self.config.max_length,
            temperature=temperature or self.config.temperature,
            top_p=top_p or self.config.top_p,
        )

        generator = og.Generator(self._model, self._tokenizer, params)
        generator.append_tokens(self._tokenizer.encode(prompt))

        start = time.perf_counter()
        try:
            # Simple greedy loop; generator handles sampling via search options.
            while not generator.is_done():
                generator.compute_logits()
                generator.generate_next_token()
            output = self._tokenizer.decode(generator.get_sequence(0))
        except Exception as exc:  # pragma: no cover - runtime specific
            output = f"[ONNX LLM generation failed: {exc}]"
        latency_ms = (time.perf_counter() - start) * 1000.0

        return {"output": output, "latency_ms": latency_ms}

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Dict[str, object]:
        """Chat-style generation. Messages should include role/content."""
        prompt = self._build_prompt(messages)
        return self.generate(
            prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
        )

