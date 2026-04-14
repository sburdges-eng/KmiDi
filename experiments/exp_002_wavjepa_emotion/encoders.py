"""
Frozen encoders for emotion experiment: WavJEPA, HuBERT, Wav2Vec2.
Common interface: load_encoder(name, config) -> encoder, extract(encoder, audio_16k) -> embedding vector.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, Optional, Union

# Lazy imports for heavy deps
_torch = None
_transformers = None


def _get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def _get_transformers():
    global _transformers
    if _transformers is None:
        import transformers
        _transformers = transformers
    return _transformers


def load_encoder(name: str, config: Dict[str, Any], device: Optional[str] = None):
    """
    Load a frozen encoder. name in: wavjepa, hubert_base, wav2vec2_base.
    Returns a model (and optional processor) for use with extract().
    """
    torch = _get_torch()
    transformers = _get_transformers()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"

    if name == "hubert_base":
        checkpoint = config.get("hubert_checkpoint", "facebook/hubert-base-ls960")
        model = transformers.HubertModel.from_pretrained(checkpoint)
        # HuBERT has no tokenizer; use feature extractor only for normalization
        processor = transformers.AutoFeatureExtractor.from_pretrained(checkpoint)
        model.eval()
        model.to(device)
        return {"model": model, "processor": processor, "device": device, "name": "hubert"}

    if name == "wav2vec2_base":
        checkpoint = config.get("wav2vec2_checkpoint", "facebook/wav2vec2-base")
        model = transformers.Wav2Vec2Model.from_pretrained(checkpoint)
        processor = transformers.Wav2Vec2Processor.from_pretrained(checkpoint)
        model.eval()
        model.to(device)
        return {"model": model, "processor": processor, "device": device, "name": "wav2vec2"}

    if name == "wavjepa":
        checkpoint = config.get("wavjepa_checkpoint", "labhamlet/wavjepa-base")
        try:
            # WavJEPA may be registered as a custom AutoModel or require the official repo.
            model = transformers.AutoModel.from_pretrained(checkpoint, trust_remote_code=True)
            processor = getattr(transformers, "AutoProcessor", None)
            if processor is not None:
                try:
                    proc = processor.from_pretrained(checkpoint, trust_remote_code=True)
                except Exception:
                    proc = None
            else:
                proc = None
            model.eval()
            model.to(device)
            return {"model": model, "processor": proc, "device": device, "name": "wavjepa"}
        except Exception as e:
            raise RuntimeError(
                f"WavJEPA load failed for {checkpoint}. "
                "If the model is not on Hugging Face, use the official WavJEPA repo and add a custom loader here."
            ) from e

    raise ValueError(f"Unknown encoder: {name}. Use wavjepa, hubert_base, or wav2vec2_base.")


def extract(
    encoder_dict: Dict[str, Any],
    audio_16k: np.ndarray,
    sample_rate: int = 16000,
    pool: str = "mean",
) -> np.ndarray:
    """
    Extract one embedding vector per clip. audio_16k: (samples,) float in [-1,1].
    pool: 'mean' (over time) or 'first'. Returns shape (embed_dim,).
    """
    torch = _get_torch()
    model = encoder_dict["model"]
    processor = encoder_dict.get("processor")
    device = encoder_dict["device"]
    name = encoder_dict.get("name", "")

    if processor is not None:
        if name == "hubert":
            # Feature extractor returns dict with input_values
            inputs = processor([audio_16k], sampling_rate=sample_rate, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(device)
        else:
            inputs = processor([audio_16k], sampling_rate=sample_rate, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(device)
    else:
        # Raw tensor; assume (1, T) or (T,)
        x = torch.from_numpy(audio_16k).float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        input_values = x.to(device)
        if input_values.dim() == 1:
            input_values = input_values.unsqueeze(0)

    with torch.no_grad():
        out = model(input_values)
    if hasattr(out, "last_hidden_state"):
        hidden = out.last_hidden_state
    elif hasattr(out, "get") and "last_hidden_state" in out:
        hidden = out.get("last_hidden_state")
    elif isinstance(out, tuple):
        hidden = out[0]
    else:
        hidden = out
        
    if isinstance(hidden, tuple):
        hidden = hidden[0]
    # (1, T, D) -> (T, D)
    hidden = hidden.squeeze(0).cpu().numpy()
    if pool == "mean":
        return np.mean(hidden, axis=0).astype(np.float32)
    return hidden[0].astype(np.float32)
