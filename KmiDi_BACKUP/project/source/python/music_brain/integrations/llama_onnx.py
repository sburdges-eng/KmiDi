"""
LLaMA ONNX integration (lightweight wrapper).

Goals:
- Lazy, optional dependency on onnxruntime.
- Simple JSON config loader under `models/llama_onnx.json`.
- Minimal surface: `generate(prompt, ...)` returning text, or None on failure.

This is a scaffold: plug in your tokenizer + decoding to get real completions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class LlamaOnnxConfig:
    model_name: str
    model_path: str
    provider: str = "cpu"  # cpu | coreml | cuda
    session_threads: int = 4
    intra_op_num_threads: int = 4
    inter_op_num_threads: int = 1
    ep_options: Dict[str, Any] = None
    max_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.9
    timeout_ms: int = 5000


def load_llama_config(config_path: Path) -> Optional[LlamaOnnxConfig]:
    """Load LLaMA ONNX config from JSON."""
    if not config_path.exists():
        return None
    with open(config_path, "r") as f:
        data = json.load(f)
    return LlamaOnnxConfig(
        model_name=data.get("model_name", "llama_onnx"),
        model_path=data.get("model_path", ""),
        provider=data.get("provider", "cpu"),
        session_threads=data.get("session_threads", 4),
        intra_op_num_threads=data.get("intra_op_num_threads", 4),
        inter_op_num_threads=data.get("inter_op_num_threads", 1),
        ep_options=data.get("ep_options") or {},
        max_tokens=data.get("max_tokens", 128),
        temperature=data.get("temperature", 0.8),
        top_p=data.get("top_p", 0.9),
        timeout_ms=data.get("timeout_ms", 5000),
    )


class LlamaOnnxGenerator:
    """Thin wrapper over onnxruntime session."""

    def __init__(self, config: LlamaOnnxConfig):
        self.config = config
        self.session = None
        self._load_session()

    def _load_session(self):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime not installed. Install with `pip install onnxruntime` "
                "or `onnxruntime-gpu`."
            ) from exc

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = self.config.intra_op_num_threads
        sess_opts.inter_op_num_threads = self.config.inter_op_num_threads

        providers = []
        if self.config.provider.lower() == "coreml":
            providers.append(("CoreMLExecutionProvider", self.config.ep_options or {}))
        elif self.config.provider.lower() == "cuda":
            providers.append(("CUDAExecutionProvider", self.config.ep_options or {}))
        # Always append CPU as fallback
        providers.append(("CPUExecutionProvider", {}))

        self.session = ort.InferenceSession(
            self.config.model_path,
            sess_options=sess_opts,
            providers=[p[0] for p in providers],
            provider_options=[p[1] for p in providers],
        )

    def is_available(self) -> bool:
        return self.session is not None

    def generate(
        self,
        prompt: str,
        *,
        tokenizer: Any = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Optional[str]:
        """
        Generate text. This is a scaffold:
        - If no tokenizer is provided, returns the prompt as echo (safe fallback).
        - Plug in your tokenizer and decoding loop for real results.
        """
        if not self.session:
            return None

        # Safe fallback: echo prompt to avoid breaking callers.
        if tokenizer is None:
            return f"[llama-onnx echo] {prompt}"

        # Minimal greedy decode example (requires compatible tokenizer + model graph).
        # This is intentionally conservative and may need adaptation to your model IO.
        try:
            input_ids = tokenizer.encode(prompt, return_tensors="np")
        except Exception:
            # Tokenizer not compatible; fallback to echo.
            return f"[llama-onnx echo] {prompt}"

        try:
            import numpy as np
        except ImportError:
            return f"[llama-onnx echo] {prompt}"

        max_new = max_tokens or self.config.max_tokens
        temp = temperature or self.config.temperature
        top_p_val = top_p or self.config.top_p

        # NOTE: Real LLaMA decoding requires causal mask + kv cache. This stub
        # calls a single forward and returns the argmax token to keep it safe.
        try:
            inputs = {"input_ids": np.array(input_ids, dtype=np.int64)}
            outputs = self.session.run(None, inputs)
            logits = outputs[0]  # shape: (1, seq, vocab)
            next_token = int(np.argmax(logits[0, -1]))
            decoded = tokenizer.decode([next_token])
            return decoded
        except Exception:
            # If anything fails, fallback to echo
            return f"[llama-onnx echo] {prompt}"


def default_llama_config_path() -> Path:
    """Locate default config at repo root /models/llama_onnx.json."""
    here = Path(__file__).resolve()
    # music_brain/integrations -> music_brain -> repo root
    repo_root = here.parents[2]
    return repo_root / "models" / "llama_onnx.json"


def build_llama_generator(config_path: Optional[Path] = None) -> Optional[LlamaOnnxGenerator]:
    """Factory that returns a generator if config + runtime are available."""
    cfg_path = config_path or default_llama_config_path()
    cfg = load_llama_config(cfg_path)
    if not cfg or not cfg.model_path:
        return None
    try:
        return LlamaOnnxGenerator(cfg)
    except Exception:
        return None

