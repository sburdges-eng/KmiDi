"""
Lyric ONNX inference helpers.

Provides a thin wrapper around an ONNX causal LM export (e.g., Gemma/GPT
fine-tuned for lyrics). This is intentionally minimal and defensive: if the
model or onnxruntime is missing, we surface clear errors instead of crashing
callers.
"""

from music_brain.lyrics.lyric_inference import LyricOnnxClient

__all__ = ["LyricOnnxClient"]
