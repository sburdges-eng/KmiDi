"""
Voice type classifier runtime wrapper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import json
import numpy as np


def _load_audio_mel(
    audio_path: str, duration: float = 3.0, sr: int = 22050, n_mels: int = 128
) -> np.ndarray:
    try:
        import librosa  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("librosa is required for voice classification") from exc

    y, _ = librosa.load(audio_path, sr=sr, mono=True, duration=duration)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # normalize to roughly [-1, 1]
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    # [1, 1, n_mels, time]
    return mel_db.astype(np.float32)[None, None, :, :]


class VoiceClassifier:
    """
    Inference helper for voice type labels (alto/bass/soprano/tenor).
    """

    def __init__(
        self,
        model_name: str = "voice_type_classifier",
        class_names: Optional[List[str]] = None,
    ):
        self.model_name = model_name
        self.class_names = class_names or ["alto", "bass", "soprano", "tenor"]
        self._engine = None
        self._try_load_engine()

    def _try_load_engine(self) -> None:
        # Support both historical import paths present in this repo.
        create_engine_by_name = None
        registry = None
        for import_target in (
            "music_brain.penta_core.ml.inference",
            "penta_core.ml.inference",
        ):
            try:
                mod = __import__(import_target, fromlist=["create_engine_by_name"])
                create_engine_by_name = getattr(mod, "create_engine_by_name")
                break
            except Exception:
                continue
        for import_target in (
            "music_brain.penta_core.ml.model_registry",
            "penta_core.ml.model_registry",
        ):
            try:
                mod = __import__(import_target, fromlist=["get_registry"])
                registry = getattr(mod, "get_registry")()
                break
            except Exception:
                continue

        if registry is not None:
            registry.discover()
        if create_engine_by_name is not None:
            self._engine = create_engine_by_name(self.model_name)
            if self._engine is not None:
                self._engine.load()

    def load_class_names_from_manifest(self, manifest_path: str) -> None:
        data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        names = data.get("class_names")
        if isinstance(names, list) and names:
            self.class_names = [str(x) for x in names]

    def classify(self, audio_path: str, top_k: int = 3) -> Dict[str, object]:
        if self._engine is None:
            raise RuntimeError(f"Model engine not available for '{self.model_name}'")

        x = _load_audio_mel(audio_path)
        result = self._engine.infer({"input": x})
        output = result.get_output().reshape(-1)
        probs = np.exp(output - np.max(output))
        probs = probs / (np.sum(probs) + 1e-8)
        idx = np.argsort(probs)[::-1][: max(1, top_k)]
        ranked = [
            {
                "label": self.class_names[i] if i < len(self.class_names) else str(i),
                "score": float(probs[i]),
            }  # noqa: E501
            for i in idx
        ]
        return {"top": ranked[0], "ranked": ranked, "latency_ms": result.latency_ms}
