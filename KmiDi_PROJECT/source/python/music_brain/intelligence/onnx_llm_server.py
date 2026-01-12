"""
FastAPI server exposing ONNX LLM generation for production deployments.

Designed to run inside the `daiw-llm-onnx` Docker service.
"""

from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from music_brain.intelligence.onnx_llm import OnnxGenAILLM, OnnxLLMConfig


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None


class PromptRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None


app = FastAPI(title="DAiW ONNX LLM", version="0.1.0")

_llm: Optional[OnnxGenAILLM] = None


@app.on_event("startup")
def startup_event() -> None:
    """Load the ONNX model on startup."""
    global _llm
    _llm = OnnxGenAILLM(OnnxLLMConfig.from_env())


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    if _llm and _llm.is_available:
        return {"status": "ok", "model_path": _llm.config.model_path}
    return {"status": "unavailable"}


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    """Chat-style endpoint taking role/content messages."""
    if not _llm or not _llm.is_available:
        raise HTTPException(status_code=503, detail="ONNX LLM not available")

    messages = [msg.dict() for msg in request.messages]
    return _llm.chat(
        messages,
        max_length=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )


@app.post("/generate")
def generate(request: PromptRequest) -> dict:
    """Single prompt generation endpoint."""
    if not _llm or not _llm.is_available:
        raise HTTPException(status_code=503, detail="ONNX LLM not available")

    return _llm.generate(
        request.prompt,
        max_length=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )

