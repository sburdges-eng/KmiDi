"""
ML Pipeline - Bridge between Python emotion system and C++ MLInterface.

This module provides a Pythonic wrapper around the penta_core_native.ml bindings,
enabling real-time emotion-driven automation via the trained (or stub) neural models.

Architecture:
    EmotionalState → feature vector → MLInterface → inference result → parameters

Models (see models/registry.json):
    - EmotionRecognizer: Audio features → emotion embedding (64-dim)
    - MelodyTransformer: Emotion embedding → note probabilities
    - HarmonyPredictor: Context → chord/harmony predictions
    - DynamicsEngine: Emotion → expression parameters (velocity, dynamics)
    - GroovePredictor: Emotion → groove/timing parameters

Usage:
    from music_brain.agents.ml_pipeline import MLPipeline

    pipeline = MLPipeline()
    pipeline.load_registry()
    pipeline.start()

    # Submit emotion features
    request_id = pipeline.submit_emotion(emotion_features)

    # Poll for results
    result = pipeline.poll()
    if result:
        dynamics = result.dynamics_params
        groove = result.groove_params
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Default model directory (relative to repo root)
DEFAULT_MODELS_DIR = Path(__file__).parent.parent.parent / "models"
DEFAULT_REGISTRY = DEFAULT_MODELS_DIR / "registry.json"


class ModelType(Enum):
    """Python enum mirroring C++ penta::ml::ModelType."""

    ChordPredictor = "ChordPredictor"
    GrooveTransfer = "GrooveTransfer"
    KeyDetector = "KeyDetector"
    IntentMapper = "IntentMapper"
    EmotionRecognizer = "EmotionRecognizer"
    MelodyTransformer = "MelodyTransformer"
    HarmonyPredictor = "HarmonyPredictor"
    DynamicsEngine = "DynamicsEngine"
    GroovePredictor = "GroovePredictor"
    Custom = "Custom"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class EmotionFeatures:
    """
    Feature vector for emotion recognition.

    This is the input to the EmotionRecognizer model. In production,
    these would come from audio analysis (spectral, rhythmic, harmonic features).
    For now, we derive them from the existing EmotionalState.
    """

    valence: float = 0.0  # -1 to 1
    arousal: float = 0.5  # 0 to 1
    tension: float = 0.0  # 0 to 1
    density: float = 0.5  # 0 to 1
    brightness: float = 0.5  # 0 to 1 (spectral centroid proxy)
    roughness: float = 0.0  # 0 to 1 (dissonance proxy)
    tempo_factor: float = 0.5  # 0 to 1 (normalized tempo)
    attack: float = 0.5  # 0 to 1 (transient sharpness)

    # Extended features for full 128-dim input
    spectral_features: List[float] = field(default_factory=lambda: [0.0] * 32)
    rhythm_features: List[float] = field(default_factory=lambda: [0.0] * 32)
    harmonic_features: List[float] = field(default_factory=lambda: [0.0] * 48)

    def to_vector(self) -> np.ndarray:
        """Convert to 128-dim feature vector for ML input."""
        base = [
            self.valence,
            self.arousal,
            self.tension,
            self.density,
            self.brightness,
            self.roughness,
            self.tempo_factor,
            self.attack,
        ]
        # Pad base features to 16 dims
        base_padded = base + [0.0] * (16 - len(base))

        return np.array(
            base_padded + self.spectral_features + self.rhythm_features + self.harmonic_features,
            dtype=np.float32,
        )


@dataclass
class DynamicsResult:
    """Result from DynamicsEngine inference."""

    velocity_curve: List[float] = field(default_factory=list)
    attack_emphasis: float = 0.5
    release_factor: float = 0.5
    dynamic_range: float = 0.6
    compression_suggestion: float = 0.3
    automation_points: List[Tuple[float, float]] = field(default_factory=list)
    confidence: float = 0.0

    @classmethod
    def from_output(cls, output: List[float], confidence: float = 0.0) -> DynamicsResult:
        """Parse ML output into structured result."""
        if len(output) < 16:
            output = output + [0.0] * (16 - len(output))

        return cls(
            velocity_curve=output[:8],
            attack_emphasis=output[8],
            release_factor=output[9],
            dynamic_range=output[10],
            compression_suggestion=output[11],
            automation_points=[(i / 8, output[i]) for i in range(8)],
            confidence=confidence,
        )


@dataclass
class GrooveResult:
    """Result from GroovePredictor inference."""

    swing_amount: float = 0.0
    timing_deviation_ms: float = 0.0
    velocity_variation: float = 0.0
    ghost_note_probability: float = 0.0
    accent_pattern: List[float] = field(default_factory=list)
    humanization_curve: List[float] = field(default_factory=list)
    confidence: float = 0.0

    @classmethod
    def from_output(cls, output: List[float], confidence: float = 0.0) -> GrooveResult:
        """Parse ML output into structured result."""
        if len(output) < 32:
            output = output + [0.0] * (32 - len(output))

        return cls(
            swing_amount=output[0],
            timing_deviation_ms=output[1] * 20.0,  # Scale to 0-20ms
            velocity_variation=output[2],
            ghost_note_probability=output[3],
            accent_pattern=output[4:12],
            humanization_curve=output[12:28],
            confidence=confidence,
        )


@dataclass
class EmotionEmbedding:
    """64-dim emotion embedding from EmotionRecognizer."""

    embedding: np.ndarray = field(default_factory=lambda: np.zeros(64, dtype=np.float32))
    primary_emotion: str = "neutral"
    confidence: float = 0.0

    @classmethod
    def from_output(cls, output: List[float], confidence: float = 0.0) -> EmotionEmbedding:
        """Parse ML output into structured result."""
        embedding = np.array(output[:64], dtype=np.float32) if len(output) >= 64 else np.zeros(64, dtype=np.float32)

        # Determine primary emotion from embedding (simplified)
        # In production, this would use a classifier head
        emotion_map = ["neutral", "grief", "anxiety", "anger", "calm", "hope", "nostalgia", "tension"]
        if len(output) > 0:
            idx = int(np.argmax(embedding[:8])) % len(emotion_map)
            primary = emotion_map[idx]
        else:
            primary = "neutral"

        return cls(embedding=embedding, primary_emotion=primary, confidence=confidence)


@dataclass
class HarmonyResult:
    """Result from HarmonyPredictor inference."""

    chord_probabilities: List[float] = field(default_factory=list)
    suggested_root: int = 0  # 0-11 pitch class
    suggested_quality: str = "major"
    voice_leading_suggestions: List[int] = field(default_factory=list)
    confidence: float = 0.0

    @classmethod
    def from_output(cls, output: List[float], confidence: float = 0.0) -> HarmonyResult:
        """Parse ML output into structured result."""
        if len(output) < 64:
            output = output + [0.0] * (64 - len(output))

        # First 12 values are pitch class probabilities
        chord_probs = output[:12]
        suggested_root = int(np.argmax(chord_probs)) if chord_probs else 0

        # Next 12 are quality indicators
        qualities = ["major", "minor", "dim", "aug", "dom7", "maj7", "min7", "dim7"]
        quality_idx = int(np.argmax(output[12:20])) % len(qualities)

        return cls(
            chord_probabilities=chord_probs,
            suggested_root=suggested_root,
            suggested_quality=qualities[quality_idx],
            voice_leading_suggestions=list(map(int, output[32:40])),
            confidence=confidence,
        )


@dataclass
class MLInferenceResult:
    """Aggregated result from multiple ML models."""

    request_id: int = 0
    emotion_embedding: Optional[EmotionEmbedding] = None
    dynamics: Optional[DynamicsResult] = None
    groove: Optional[GrooveResult] = None
    harmony: Optional[HarmonyResult] = None
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


# =============================================================================
# ML Pipeline
# =============================================================================


class MLPipeline:
    """
    Bridge between Python emotion system and C++ MLInterface.

    Provides high-level methods for:
    - Loading models from registry
    - Submitting inference requests
    - Polling for results
    - Converting between Python types and C++ structs
    """

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        use_gpu: bool = False,
        use_coreml: bool = True,
    ):
        self.models_dir = models_dir or DEFAULT_MODELS_DIR
        self.registry_path = self.models_dir / "registry.json"
        self.use_gpu = use_gpu
        self.use_coreml = use_coreml

        # C++ interface (lazy loaded)
        self._ml_module: Optional[Any] = None
        self._interface: Optional[Any] = None
        self._native_available = False

        # Request tracking
        self._pending_requests: Dict[int, Tuple[ModelType, float]] = {}
        self._results: Dict[int, Dict[str, Any]] = {}
        self._lock = threading.Lock()

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {}

        # Try to import native module
        self._try_import_native()

    def _try_import_native(self) -> bool:
        """Attempt to import penta_core_native.ml module."""
        try:
            import penta_core_native as pcn

            self._ml_module = pcn.ml
            self._native_available = True
            logger.info("penta_core_native.ml module loaded successfully")
            return True
        except ImportError as e:
            logger.warning(f"penta_core_native not available: {e}")
            logger.info("ML pipeline will use fallback heuristics")
            self._native_available = False
            return False

    @property
    def native_available(self) -> bool:
        """Check if C++ native bindings are available."""
        return self._native_available

    @property
    def is_running(self) -> bool:
        """Check if inference thread is running."""
        if self._interface is None:
            return False
        return self._interface.is_running()

    def load_registry(self, registry_path: Optional[Path] = None) -> bool:
        """
        Load all models from registry.json.

        Returns:
            True if all models loaded successfully
        """
        path = registry_path or self.registry_path

        if not path.exists():
            logger.error(f"Registry not found: {path}")
            return False

        if not self._native_available:
            logger.info("Native module not available, loading registry for metadata only")
            with open(path) as f:
                self._registry = json.load(f)
            return True

        # Initialize interface if needed
        if self._interface is None:
            config = self._ml_module.MLConfig()
            config.model_directory = str(self.models_dir)
            config.use_gpu = self.use_gpu
            config.use_coreml = self.use_coreml
            self._interface = self._ml_module.MLInterface(config)

        # Load registry via C++ interface
        success = self._interface.load_registry(str(path))

        if success:
            logger.info(f"Loaded models from registry: {path}")
        else:
            logger.warning(f"Some models failed to load from: {path}")

        return success

    def start(self) -> bool:
        """Start the inference thread."""
        if not self._native_available:
            logger.info("Running in fallback mode (no native inference)")
            return True

        if self._interface is None:
            logger.error("Must call load_registry() before start()")
            return False

        success = self._interface.start()
        if success:
            logger.info("ML inference thread started")
        return success

    def stop(self) -> None:
        """Stop the inference thread."""
        if self._interface is not None:
            self._interface.stop()
            logger.info("ML inference thread stopped")

    # =========================================================================
    # Inference Methods
    # =========================================================================

    def submit_emotion(
        self,
        features: EmotionFeatures,
        timestamp: int = 0,
    ) -> Optional[int]:
        """
        Submit emotion features for recognition.

        Returns:
            Request ID if queued, None if failed
        """
        return self._submit_features(
            ModelType.EmotionRecognizer,
            features.to_vector(),
            timestamp,
        )

    def submit_dynamics(
        self,
        emotion_embedding: np.ndarray,
        timestamp: int = 0,
    ) -> Optional[int]:
        """
        Submit emotion embedding for dynamics prediction.

        Args:
            emotion_embedding: 64-dim emotion embedding
            timestamp: Audio sample timestamp

        Returns:
            Request ID if queued, None if failed
        """
        # DynamicsEngine expects 32-dim input (we use first 32 of embedding)
        features = emotion_embedding[:32] if len(emotion_embedding) >= 32 else np.zeros(32, dtype=np.float32)
        return self._submit_features(ModelType.DynamicsEngine, features, timestamp)

    def submit_groove(
        self,
        emotion_embedding: np.ndarray,
        timestamp: int = 0,
    ) -> Optional[int]:
        """
        Submit emotion embedding for groove prediction.

        Args:
            emotion_embedding: 64-dim emotion embedding
            timestamp: Audio sample timestamp

        Returns:
            Request ID if queued, None if failed
        """
        # GroovePredictor expects 64-dim input
        features = emotion_embedding if len(emotion_embedding) >= 64 else np.zeros(64, dtype=np.float32)
        return self._submit_features(ModelType.GroovePredictor, features, timestamp)

    def submit_harmony(
        self,
        context_features: np.ndarray,
        timestamp: int = 0,
    ) -> Optional[int]:
        """
        Submit context features for harmony prediction.

        Args:
            context_features: 128-dim context (emotion + pitch history)
            timestamp: Audio sample timestamp

        Returns:
            Request ID if queued, None if failed
        """
        return self._submit_features(ModelType.HarmonyPredictor, context_features, timestamp)

    def _submit_features(
        self,
        model_type: ModelType,
        features: np.ndarray,
        timestamp: int,
    ) -> Optional[int]:
        """Internal method to submit features to ML interface."""
        if not self._native_available:
            # Generate fallback result immediately
            return self._generate_fallback_result(model_type, features)

        if self._interface is None or not self.is_running:
            logger.warning("ML interface not running")
            return None

        # Convert Python ModelType to C++ enum
        native_type = getattr(self._ml_module.ModelType, model_type.value)

        # Submit to C++ interface
        queued, request_id = self._interface.submit_features(native_type, features, timestamp)

        if queued:
            with self._lock:
                self._pending_requests[request_id] = (model_type, time.time())
            return request_id
        else:
            logger.warning(f"Request queue full for {model_type.value}")
            return None

    def _generate_fallback_result(
        self,
        model_type: ModelType,
        features: np.ndarray,
    ) -> int:
        """Generate heuristic-based fallback result when native is unavailable."""
        request_id = hash((model_type.value, time.time())) & 0xFFFFFFFF

        # Simple heuristic fallbacks
        if model_type == ModelType.EmotionRecognizer:
            # Echo first 64 features as "embedding"
            output = list(features[:64]) if len(features) >= 64 else list(features) + [0.0] * (64 - len(features))
            result = {"model_type": model_type, "output": output, "confidence": 0.5, "success": True}
        elif model_type == ModelType.DynamicsEngine:
            # Generate default dynamics
            output = [0.5] * 16
            output[10] = 0.6  # dynamic_range
            result = {"model_type": model_type, "output": output, "confidence": 0.3, "success": True}
        elif model_type == ModelType.GroovePredictor:
            # Generate default groove
            output = [0.52, 5.0 / 20.0, 0.15, 0.05] + [0.5] * 28
            result = {"model_type": model_type, "output": output, "confidence": 0.3, "success": True}
        elif model_type == ModelType.HarmonyPredictor:
            # Generate default harmony (C major)
            output = [1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.8, 0.0, 0.3, 0.0, 0.0] + [0.0] * 52
            result = {"model_type": model_type, "output": output, "confidence": 0.3, "success": True}
        else:
            output = [0.0] * 128
            result = {"model_type": model_type, "output": output, "confidence": 0.0, "success": False}

        result["request_id"] = request_id
        result["latency_ms"] = 0.0

        with self._lock:
            self._results[request_id] = result

        return request_id

    def poll(self) -> Optional[Dict[str, Any]]:
        """
        Poll for next inference result.

        Returns:
            Result dict or None if no results available
        """
        # Check fallback results first
        with self._lock:
            if self._results:
                request_id = next(iter(self._results))
                return self._results.pop(request_id)

        if not self._native_available or self._interface is None:
            return None

        result = self._interface.poll_result()

        if result is not None:
            with self._lock:
                self._pending_requests.pop(result.get("request_id", 0), None)
            self._trigger_callback("result", result)

        return result

    def poll_typed(self) -> Optional[MLInferenceResult]:
        """
        Poll for result and parse into typed dataclass.

        Returns:
            MLInferenceResult or None if no results
        """
        raw = self.poll()
        if raw is None:
            return None

        return self._parse_result(raw)

    def _parse_result(self, raw: Dict[str, Any]) -> MLInferenceResult:
        """Parse raw result dict into typed MLInferenceResult."""
        result = MLInferenceResult(
            request_id=raw.get("request_id", 0),
            latency_ms=raw.get("latency_ms", 0.0),
        )

        model_type = raw.get("model_type")
        output = raw.get("output", [])
        confidence = raw.get("confidence", 0.0)

        if isinstance(model_type, ModelType):
            type_str = model_type.value
        elif hasattr(model_type, "name"):
            type_str = model_type.name
        else:
            type_str = str(model_type)

        if type_str == "EmotionRecognizer":
            result.emotion_embedding = EmotionEmbedding.from_output(output, confidence)
        elif type_str == "DynamicsEngine":
            result.dynamics = DynamicsResult.from_output(output, confidence)
        elif type_str == "GroovePredictor":
            result.groove = GrooveResult.from_output(output, confidence)
        elif type_str == "HarmonyPredictor":
            result.harmony = HarmonyResult.from_output(output, confidence)

        return result

    # =========================================================================
    # Batch Inference
    # =========================================================================

    def process_emotion_full(
        self,
        features: EmotionFeatures,
        timeout_ms: float = 100.0,
    ) -> MLInferenceResult:
        """
        Run full emotion processing pipeline synchronously.

        Submits to EmotionRecognizer, then uses embedding for Dynamics and Groove.

        Args:
            features: Emotion features to process
            timeout_ms: Max time to wait for results

        Returns:
            Aggregated MLInferenceResult
        """
        result = MLInferenceResult()
        deadline = time.time() + timeout_ms / 1000.0

        # Step 1: Get emotion embedding
        req_id = self.submit_emotion(features)
        if req_id is None:
            return result

        while time.time() < deadline:
            polled = self.poll_typed()
            if polled and polled.emotion_embedding:
                result.emotion_embedding = polled.emotion_embedding
                result.request_id = polled.request_id
                break
            time.sleep(0.001)

        if result.emotion_embedding is None:
            return result

        # Step 2: Submit dynamics and groove in parallel
        embedding = result.emotion_embedding.embedding
        dynamics_id = self.submit_dynamics(embedding)
        groove_id = self.submit_groove(embedding)

        # Collect results
        pending = {dynamics_id, groove_id} - {None}
        while pending and time.time() < deadline:
            polled = self.poll_typed()
            if polled:
                if polled.dynamics:
                    result.dynamics = polled.dynamics
                    pending.discard(dynamics_id)
                if polled.groove:
                    result.groove = polled.groove
                    pending.discard(groove_id)
            time.sleep(0.001)

        result.latency_ms = (time.time() - deadline + timeout_ms / 1000.0) * 1000.0
        return result

    # =========================================================================
    # Callbacks
    # =========================================================================

    def on(self, event: str, callback: Callable) -> None:
        """Register callback for event."""
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)

    def off(self, event: str, callback: Optional[Callable] = None) -> None:
        """Remove callback(s)."""
        if callback:
            callbacks = self._callbacks.get(event, [])
            if callback in callbacks:
                callbacks.remove(callback)
            else:
                logger.warning("Callback not found for event '%s'", event)
        else:
            self._callbacks.pop(event, None)

    def _trigger_callback(self, event: str, data: Any = None) -> None:
        """Trigger callbacks for event."""
        for cb in self._callbacks.get(event, []):
            try:
                cb(data)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        if not self._native_available or self._interface is None:
            return {
                "native_available": False,
                "running": False,
                "pending_requests": len(self._pending_requests),
            }

        stats = self._interface.get_stats()
        return {
            "native_available": True,
            "running": self.is_running,
            "total_requests": stats.total_requests,
            "completed_requests": stats.completed_requests,
            "failed_requests": stats.failed_requests,
            "queue_overflows": stats.queue_overflows,
            "avg_latency_ms": stats.avg_latency_ms,
            "max_latency_ms": stats.max_latency_ms,
            "pending_requests": len(self._pending_requests),
        }

    # =========================================================================
    # Context Manager
    # =========================================================================

    def __enter__(self) -> MLPipeline:
        self.load_registry()
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

