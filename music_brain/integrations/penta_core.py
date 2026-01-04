"""
Penta-Core Integration Module.

This module provides the interface for integrating DAiW-Music-Brain
with the penta-core system (https://github.com/sburdges-eng/penta-core).

The integration follows DAiW-Music-Brain's core philosophy:
"Interrogate Before Generate" - emotional intent drives technical decisions.

Usage:
    from music_brain.integrations.penta_core import PentaCoreIntegration

    integration = PentaCoreIntegration()

    # Send song intent to penta-core
    result = integration.send_intent(complete_song_intent)

    # Check connection status
    if integration.is_connected():
        suggestions = integration.receive_suggestions()
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import json
import logging
import time

logger = logging.getLogger(__name__)

# Try to import HTTP libraries
try:
    import urllib.request
    import urllib.error
    HAS_URLLIB = True
except ImportError:
    HAS_URLLIB = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


@dataclass
class PentaCoreConfig:
    """Configuration for penta-core integration.

    Attributes:
        endpoint_url: The URL of the penta-core service endpoint.
        api_key: Optional API key for authentication.
        timeout_seconds: Request timeout in seconds.
        verify_ssl: Whether to verify SSL certificates.
    """

    endpoint_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout_seconds: int = 30
    verify_ssl: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            "endpoint_url": self.endpoint_url,
            "api_key": self.api_key,
            "timeout_seconds": self.timeout_seconds,
            "verify_ssl": self.verify_ssl,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PentaCoreConfig":
        """Create configuration from dictionary."""
        return cls(
            endpoint_url=data.get("endpoint_url"),
            api_key=data.get("api_key"),
            timeout_seconds=data.get("timeout_seconds", 30),
            verify_ssl=data.get("verify_ssl", True),
        )


class PentaCoreIntegration:
    """Integration interface for penta-core system.

    This class provides methods for communicating with the penta-core
    service via HTTP, enabling data exchange while preserving emotional intent
    context from DAiW-Music-Brain's three-phase intent schema.

    The integration supports:
    - Sending song intents (Phase 0, 1, 2 data)
    - Sending groove templates
    - Sending chord progression analysis
    - Receiving suggestions and feedback

    Example:
        >>> from music_brain.integrations.penta_core import PentaCoreIntegration, PentaCoreConfig
        >>> config = PentaCoreConfig(endpoint_url="http://localhost:8000")
        >>> integration = PentaCoreIntegration(config=config)
        >>> if integration.connect():
        ...     result = integration.send_intent(song_intent)
    """

    def __init__(self, config: Optional[PentaCoreConfig] = None):
        """Initialize the penta-core integration.

        Args:
            config: Optional configuration for the integration.
                    If not provided, defaults will be used.
        """
        self._config = config or PentaCoreConfig()
        self._connected = False
        self._session_id: Optional[str] = None
        self._last_request_time: float = 0
        self._request_queue: List[Dict[str, Any]] = []

    @property
    def config(self) -> PentaCoreConfig:
        """Get the current configuration."""
        return self._config

    def is_connected(self) -> bool:
        """Check if the integration is connected to penta-core.

        Returns:
            True if connected and authenticated, False otherwise.
        """
        return self._connected and self._config.endpoint_url is not None

    def _make_request(
        self,
        endpoint: str,
        method: str = "POST",
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make an HTTP request to the penta-core service.

        Args:
            endpoint: API endpoint path (e.g., "/api/intent")
            method: HTTP method (GET, POST, PUT, DELETE)
            data: Request body data (for POST/PUT)

        Returns:
            Response data as dictionary

        Raises:
            ConnectionError: If request fails
        """
        url = f"{self._config.endpoint_url.rstrip('/')}{endpoint}"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"

        if self._session_id:
            headers["X-Session-ID"] = self._session_id

        try:
            if HAS_REQUESTS:
                return self._make_request_requests(url, method, headers, data)
            elif HAS_URLLIB:
                return self._make_request_urllib(url, method, headers, data)
            else:
                raise ConnectionError("No HTTP library available (install requests)")
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise ConnectionError(f"Request to {url} failed: {e}")

    def _make_request_requests(
        self,
        url: str,
        method: str,
        headers: Dict[str, str],
        data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Make request using requests library."""
        kwargs = {
            "headers": headers,
            "timeout": self._config.timeout_seconds,
            "verify": self._config.verify_ssl,
        }

        if data is not None:
            kwargs["json"] = data

        response = requests.request(method, url, **kwargs)
        response.raise_for_status()

        self._last_request_time = time.time()
        return response.json()

    def _make_request_urllib(
        self,
        url: str,
        method: str,
        headers: Dict[str, str],
        data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Make request using urllib."""
        body = None
        if data is not None:
            body = json.dumps(data).encode("utf-8")

        request = urllib.request.Request(url, data=body, headers=headers, method=method)

        try:
            with urllib.request.urlopen(
                request,
                timeout=self._config.timeout_seconds,
            ) as response:
                self._last_request_time = time.time()
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise ConnectionError(f"HTTP {e.code}: {e.reason}")
        except urllib.error.URLError as e:
            raise ConnectionError(f"URL Error: {e.reason}")

    def connect(self) -> bool:
        """Establish connection to penta-core service.

        Returns:
            True if connection was successful, False otherwise.

        Raises:
            ValueError: If endpoint_url is not configured.
        """
        if not self._config.endpoint_url:
            raise ValueError(
                "Cannot connect: endpoint_url not configured. "
                "Set config.endpoint_url before calling connect()."
            )

        try:
            # Ping the health endpoint
            response = self._make_request("/api/health", method="GET")

            if response.get("status") == "ok":
                self._connected = True
                self._session_id = response.get("session_id")
                logger.info(f"Connected to penta-core at {self._config.endpoint_url}")
                return True

            logger.warning(f"Penta-core health check returned: {response}")
            return False

        except ConnectionError as e:
            logger.warning(f"Failed to connect to penta-core: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from penta-core service."""
        if self._connected and self._session_id:
            try:
                self._make_request("/api/session/close", method="POST", data={
                    "session_id": self._session_id,
                })
            except ConnectionError:
                pass  # Ignore errors during disconnect

        self._connected = False
        self._session_id = None
        logger.info("Disconnected from penta-core")

    def _serialize_intent(self, intent: Any) -> Dict[str, Any]:
        """Serialize an intent object to dictionary."""
        if hasattr(intent, "to_dict"):
            return intent.to_dict()
        elif hasattr(intent, "__dict__"):
            return {k: v for k, v in intent.__dict__.items() if not k.startswith("_")}
        elif isinstance(intent, dict):
            return intent
        else:
            return {"data": str(intent)}

    def send_intent(self, intent: Any) -> Dict[str, Any]:
        """Send a song intent to penta-core.

        Sends the complete song intent (Phase 0, 1, 2 data) to penta-core
        for processing. The emotional context from Phase 0 is preserved
        to ensure that any suggestions returned align with the creator's
        core wound/desire.

        Args:
            intent: A CompleteSongIntent object or compatible dict
                   containing the three-phase intent data.

        Returns:
            A dictionary containing the response from penta-core,
            including any processing status or immediate feedback.

        Raises:
            ConnectionError: If not connected to penta-core.
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to penta-core. Call connect() first.")

        intent_data = self._serialize_intent(intent)

        # Ensure emotional context is preserved
        payload = {
            "type": "song_intent",
            "intent": intent_data,
            "preserve_emotional_context": True,
            "timestamp": time.time(),
        }

        response = self._make_request("/api/intent", method="POST", data=payload)

        return {
            "status": response.get("status", "received"),
            "intent_id": response.get("intent_id"),
            "processing_state": response.get("processing_state"),
            "suggestions": response.get("suggestions", []),
            "message": response.get("message", "Intent received"),
        }

    def send_groove(self, groove_template: Any) -> Dict[str, Any]:
        """Send a groove template to penta-core.

        Sends extracted groove data for processing or storage.

        Args:
            groove_template: A GrooveTemplate object or compatible dict.

        Returns:
            A dictionary containing the response from penta-core.

        Raises:
            ConnectionError: If not connected to penta-core.
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to penta-core. Call connect() first.")

        # Serialize groove template
        if hasattr(groove_template, "to_dict"):
            groove_data = groove_template.to_dict()
        elif isinstance(groove_template, dict):
            groove_data = groove_template
        else:
            groove_data = {"template": str(groove_template)}

        payload = {
            "type": "groove_template",
            "groove": groove_data,
            "timestamp": time.time(),
        }

        response = self._make_request("/api/groove", method="POST", data=payload)

        return {
            "status": response.get("status", "received"),
            "groove_id": response.get("groove_id"),
            "analysis": response.get("analysis", {}),
            "style_detected": response.get("style_detected"),
            "message": response.get("message", "Groove received"),
        }

    def send_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Send chord progression analysis to penta-core.

        Sends analysis results including emotional character,
        rule breaks, and suggestions.

        Args:
            analysis: A dictionary containing progression analysis data.

        Returns:
            A dictionary containing the response from penta-core.

        Raises:
            ConnectionError: If not connected to penta-core.
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to penta-core. Call connect() first.")

        payload = {
            "type": "progression_analysis",
            "analysis": analysis,
            "timestamp": time.time(),
        }

        response = self._make_request("/api/analysis", method="POST", data=payload)

        return {
            "status": response.get("status", "received"),
            "analysis_id": response.get("analysis_id"),
            "validation": response.get("validation", {}),
            "enhancements": response.get("enhancements", []),
            "message": response.get("message", "Analysis received"),
        }

    def receive_suggestions(self) -> List[str]:
        """Receive creative suggestions from penta-core.

        Retrieves suggestions that have been generated based on
        previously sent intents or analysis data.

        Returns:
            A list of suggestion strings.

        Raises:
            ConnectionError: If not connected to penta-core.
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to penta-core. Call connect() first.")

        response = self._make_request("/api/suggestions", method="GET")

        suggestions = response.get("suggestions", [])

        # Extract text from suggestion objects if needed
        result = []
        for s in suggestions:
            if isinstance(s, str):
                result.append(s)
            elif isinstance(s, dict):
                result.append(s.get("text", s.get("suggestion", str(s))))
            else:
                result.append(str(s))

        return result

    def receive_feedback(self) -> Dict[str, Any]:
        """Receive processing feedback from penta-core.

        Retrieves feedback on previously sent data, including
        validation results and processing status.

        Returns:
            A dictionary containing feedback data.

        Raises:
            ConnectionError: If not connected to penta-core.
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to penta-core. Call connect() first.")

        response = self._make_request("/api/feedback", method="GET")

        return {
            "status": response.get("status", "ok"),
            "feedback": response.get("feedback", {}),
            "processing_complete": response.get("processing_complete", True),
            "errors": response.get("errors", []),
            "warnings": response.get("warnings", []),
        }

    def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status.

        Returns:
            Dictionary with processing queue status and active jobs.
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to penta-core. Call connect() first.")

        response = self._make_request("/api/status", method="GET")

        return {
            "queue_length": response.get("queue_length", 0),
            "active_jobs": response.get("active_jobs", []),
            "completed_jobs": response.get("completed_jobs", []),
            "server_status": response.get("server_status", "unknown"),
        }


# =============================================================================
# ENHANCED: Local Penta-Core Integration
# =============================================================================

class LocalPentaCoreIntegration:
    """
    Local integration with Penta-Core Python bindings.

    Unlike PentaCoreIntegration (which uses HTTP), this class
    integrates directly with the local Python bindings for
    real-time performance.

    Usage:
        from music_brain.integrations.penta_core import LocalPentaCoreIntegration

        integration = LocalPentaCoreIntegration()

        # Process audio and get analysis
        result = integration.process_audio(audio_buffer)

        # Get harmony analysis
        chord = integration.get_current_chord()
        scale = integration.get_current_scale()

        # Apply dynamics from emotion
        dynamics = integration.get_dynamics_for_emotion("melancholy")
    """

    def __init__(self, sample_rate: float = 48000.0):
        """
        Initialize local Penta-Core integration.

        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self._harmony_engine = None
        self._groove_engine = None
        self._dynamics_integration = None
        self._ml_interface = None

        self._initialize_engines()

    def _initialize_engines(self):
        """Initialize Penta-Core engines."""
        # Try to import local dynamics integration
        try:
            from .dynamics_integration import DynamicsIntegration
            self._dynamics_integration = DynamicsIntegration()
        except ImportError:
            pass

        # Try to import C++ bindings
        try:
            from penta_core import HarmonyEngine, GrooveEngine
            self._harmony_engine = HarmonyEngine(self.sample_rate)
            self._groove_engine = GrooveEngine(self.sample_rate)
        except ImportError:
            pass

        # Try to import ML interface
        try:
            from penta_core.ml import get_registry
            self._ml_interface = get_registry()
        except ImportError:
            pass

    # =========================================================================
    # Harmony Integration
    # =========================================================================

    def process_notes(self, notes: List[tuple]) -> Dict[str, Any]:
        """
        Process MIDI notes through harmony engine.

        Args:
            notes: List of (pitch, velocity) tuples

        Returns:
            Harmony analysis results
        """
        if self._harmony_engine:
            # Native processing
            self._harmony_engine.process_notes(notes)
            return {
                "chord": self._harmony_engine.get_current_chord(),
                "scale": self._harmony_engine.get_current_scale(),
            }
        else:
            # Fallback: basic pitch class analysis
            pitch_classes = [0] * 12
            for pitch, velocity in notes:
                if velocity > 0:
                    pitch_classes[pitch % 12] += velocity

            # Find root and quality
            max_pc = max(range(12), key=lambda x: pitch_classes[x])
            has_minor_third = pitch_classes[(max_pc + 3) % 12] > 0
            has_major_third = pitch_classes[(max_pc + 4) % 12] > 0

            quality = "minor" if has_minor_third and not has_major_third else "major"

            return {
                "chord": {"root": max_pc, "quality": quality},
                "scale": {"root": max_pc, "type": "natural_minor" if quality == "minor" else "major"},
            }

    def get_chord_suggestions(
        self,
        current_chord: Dict[str, Any],
        emotion: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get chord suggestions based on current state.

        Args:
            current_chord: Current chord info
            emotion: Optional emotion context

        Returns:
            List of suggested chords
        """
        suggestions = []

        # Basic circle of fifths suggestions
        root = current_chord.get("root", 0)
        quality = current_chord.get("quality", "major")

        # Common progressions
        if quality == "major":
            # I -> IV, V, vi
            suggestions.append({"root": (root + 5) % 12, "quality": "major"})  # IV
            suggestions.append({"root": (root + 7) % 12, "quality": "major"})  # V
            suggestions.append({"root": (root + 9) % 12, "quality": "minor"})  # vi
        else:
            # i -> iv, III, VII
            suggestions.append({"root": (root + 5) % 12, "quality": "minor"})  # iv
            suggestions.append({"root": (root + 3) % 12, "quality": "major"})  # III
            suggestions.append({"root": (root + 10) % 12, "quality": "major"})  # VII

        return suggestions

    # =========================================================================
    # Groove Integration
    # =========================================================================

    def analyze_groove(self, onset_times: List[float]) -> Dict[str, Any]:
        """
        Analyze groove from onset times.

        Args:
            onset_times: List of onset times in seconds

        Returns:
            Groove analysis results
        """
        if self._groove_engine:
            return self._groove_engine.analyze(onset_times)

        # Fallback: basic tempo estimation
        if len(onset_times) < 2:
            return {"tempo": 120.0, "swing": 0.0}

        # Calculate inter-onset intervals
        iois = [onset_times[i+1] - onset_times[i] for i in range(len(onset_times) - 1)]
        avg_ioi = sum(iois) / len(iois)

        # Estimate tempo (assuming 8th notes)
        tempo = 30.0 / avg_ioi if avg_ioi > 0 else 120.0
        tempo = max(60, min(200, tempo))

        # Estimate swing (ratio of odd to even beats)
        if len(iois) >= 4:
            odd_iois = iois[1::2]
            even_iois = iois[::2]
            if odd_iois and even_iois:
                swing = sum(odd_iois) / sum(even_iois) - 1.0
                swing = max(0, min(1, swing * 2))
            else:
                swing = 0.0
        else:
            swing = 0.0

        return {"tempo": tempo, "swing": swing}

    def apply_groove(
        self,
        notes: List[Dict[str, Any]],
        template: str = "straight",
    ) -> List[Dict[str, Any]]:
        """
        Apply groove template to notes.

        Args:
            notes: List of note dicts with 'time', 'pitch', 'velocity', 'duration'
            template: Groove template name

        Returns:
            Notes with adjusted timing
        """
        if self._groove_engine:
            return self._groove_engine.apply_template(notes, template)

        # Fallback: basic swing application
        swing_amounts = {
            "straight": 0.0,
            "light_swing": 0.05,
            "medium_swing": 0.15,
            "heavy_swing": 0.25,
            "laid_back": 0.03,
        }

        swing = swing_amounts.get(template, 0.0)

        result = []
        for note in notes:
            new_note = note.copy()
            # Apply swing to upbeats (assuming 8th note grid)
            beat_position = (note["time"] % 0.5) / 0.5
            if beat_position > 0.4:  # Upbeat
                new_note["time"] += swing * 0.5
            result.append(new_note)

        return result

    # =========================================================================
    # Dynamics Integration
    # =========================================================================

    def get_dynamics_for_emotion(
        self,
        emotion_name: str,
        section: str = "verse",
    ) -> Dict[str, Any]:
        """
        Get dynamics parameters for an emotion.

        Args:
            emotion_name: Emotion name
            section: Section type

        Returns:
            Dynamics parameters
        """
        if self._dynamics_integration:
            from .dynamics_integration import get_dynamics_for_emotion
            params = get_dynamics_for_emotion(emotion_name, section)
            return {
                "target_lufs": params.target_lufs,
                "velocity_mean": params.velocity_mean,
                "velocity_range": (params.velocity_min, params.velocity_max),
                "note_density": params.note_density,
            }

        # Fallback: basic emotion mapping
        emotion_dynamics = {
            "melancholy": {"velocity_mean": 60, "target_lufs": -18},
            "sad": {"velocity_mean": 55, "target_lufs": -20},
            "angry": {"velocity_mean": 100, "target_lufs": -10},
            "happy": {"velocity_mean": 85, "target_lufs": -12},
            "peaceful": {"velocity_mean": 50, "target_lufs": -18},
            "excited": {"velocity_mean": 95, "target_lufs": -10},
        }

        defaults = {"velocity_mean": 75, "target_lufs": -14}
        return emotion_dynamics.get(emotion_name.lower(), defaults)

    def set_section_context(
        self,
        section_type: str,
        start_bar: int,
        end_bar: int,
        emotion: Optional[Dict[str, float]] = None,
    ):
        """
        Set section context for dynamics processing.

        Args:
            section_type: Section type name
            start_bar: Starting bar
            end_bar: Ending bar
            emotion: Optional PAD emotion dict
        """
        if self._dynamics_integration:
            from .dynamics_integration import SectionType, EmotionState

            try:
                stype = SectionType(section_type.lower())
            except ValueError:
                stype = SectionType.UNKNOWN

            emo = None
            if emotion:
                emo = EmotionState(
                    valence=emotion.get("valence", 0),
                    arousal=emotion.get("arousal", 0.5),
                    dominance=emotion.get("dominance", 0.5),
                    intensity=emotion.get("intensity", 0.5),
                )

            self._dynamics_integration.add_section(stype, start_bar, end_bar, emo)

    # =========================================================================
    # ML Integration
    # =========================================================================

    def predict_next_chord(
        self,
        context: List[Dict[str, Any]],
        num_suggestions: int = 3,
        temperature: float = 1.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Use ML to predict next chord.

        Args:
            context: List of previous chords (with 'root' and 'quality' keys)
            num_suggestions: Number of alternative suggestions
            temperature: Sampling temperature (higher = more random)

        Returns:
            Predicted chord dict with alternatives, or None if unavailable
        """
        # Try ML interface first
        if self._ml_interface:
            model = self._ml_interface.get("harmony_predictor")
            if model:
                try:
                    # Encode context as chord symbols
                    chord_symbols = []
                    for chord in context[-8:]:  # Last 8 chords
                        root = chord.get("root", 0)
                        quality = chord.get("quality", "major")
                        note_names = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
                        symbol = note_names[root % 12]
                        if quality == "minor":
                            symbol += "m"
                        elif quality == "diminished":
                            symbol += "dim"
                        elif quality == "augmented":
                            symbol += "aug"
                        chord_symbols.append(symbol)

                    # Try using chord_predictor module
                    try:
                        from penta_core.ml.chord_predictor import ChordPredictor
                        predictor = ChordPredictor(use_fallback=True)
                        prediction = predictor.predict(
                            chord_symbols,
                            num_predictions=num_suggestions,
                            temperature=temperature,
                        )

                        # Convert prediction to dict format
                        return {
                            "chord": prediction.chord,
                            "root": self._chord_to_root(prediction.chord),
                            "quality": prediction.quality,
                            "confidence": prediction.confidence,
                            "alternatives": [
                                {
                                    "chord": alt[0],
                                    "confidence": alt[1],
                                }
                                for alt in prediction.alternatives
                            ],
                        }
                    except ImportError:
                        pass

                except Exception as e:
                    logger.warning(f"ML chord prediction failed: {e}")

        # Fallback: rule-based prediction using Markov-like transitions
        if not context:
            return {"chord": "C", "root": 0, "quality": "major", "confidence": 0.5}

        last_chord = context[-1]
        root = last_chord.get("root", 0)
        quality = last_chord.get("quality", "major")

        # Common chord transitions
        if quality == "major":
            transitions = [
                ({"root": (root + 7) % 12, "quality": "major"}, 0.35),  # V
                ({"root": (root + 5) % 12, "quality": "major"}, 0.30),  # IV
                ({"root": (root + 9) % 12, "quality": "minor"}, 0.20),  # vi
                ({"root": (root + 2) % 12, "quality": "minor"}, 0.15),  # ii
            ]
        else:
            transitions = [
                ({"root": (root + 3) % 12, "quality": "major"}, 0.35),  # III
                ({"root": (root + 5) % 12, "quality": "minor"}, 0.25),  # iv
                ({"root": (root + 10) % 12, "quality": "major"}, 0.25),  # VII
                ({"root": (root + 7) % 12, "quality": "major"}, 0.15),  # V
            ]

        # Return best prediction with alternatives
        best = transitions[0]
        return {
            "root": best[0]["root"],
            "quality": best[0]["quality"],
            "confidence": best[1],
            "alternatives": [
                {"root": t[0]["root"], "quality": t[0]["quality"], "confidence": t[1]}
                for t in transitions[1:num_suggestions]
            ],
        }

    def _chord_to_root(self, chord_symbol: str) -> int:
        """Convert chord symbol to root pitch class."""
        note_map = {
            "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
            "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
            "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
        }

        # Extract root from chord symbol
        if len(chord_symbol) >= 2 and chord_symbol[1] in "#b":
            root_str = chord_symbol[:2]
        else:
            root_str = chord_symbol[0]

        return note_map.get(root_str, 0)

    def predict_chord_progression(
        self,
        starting_chords: List[Dict[str, Any]],
        length: int = 4,
        temperature: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Generate a chord progression of specified length.

        Args:
            starting_chords: Initial context chords
            length: Number of chords to generate
            temperature: Sampling temperature

        Returns:
            List of predicted chords
        """
        context = list(starting_chords)
        predictions = []

        for _ in range(length):
            pred = self.predict_next_chord(context, temperature=temperature)
            if pred:
                predictions.append(pred)
                context.append({"root": pred["root"], "quality": pred["quality"]})
            else:
                break

        return predictions

    def export_training_data(self) -> Dict[str, Any]:
        """
        Export all integration data for training.

        Collects data from:
        - Dynamics integration (section/emotion/dynamics mappings)
        - Harmony analysis history
        - Groove analysis history

        Returns:
            Training-ready data with samples for each subsystem
        """
        from datetime import datetime

        data = {
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "dynamics_samples": [],
            "harmony_samples": [],
            "groove_samples": [],
            "metadata": {
                "sample_rate": self.sample_rate,
                "has_harmony_engine": self._harmony_engine is not None,
                "has_groove_engine": self._groove_engine is not None,
                "has_dynamics_integration": self._dynamics_integration is not None,
            },
        }

        # Export dynamics data
        if self._dynamics_integration:
            dynamics_data = self._dynamics_integration.export_training_data()
            data["dynamics_samples"] = dynamics_data

            # Also extract emotion-to-dynamics mappings
            data["emotion_dynamics_map"] = self._build_emotion_dynamics_map()

        # Export harmony analysis history if available
        if hasattr(self, "_harmony_history"):
            data["harmony_samples"] = self._harmony_history

        # Export groove analysis history if available
        if hasattr(self, "_groove_history"):
            data["groove_samples"] = self._groove_history

        # Add statistics
        data["statistics"] = {
            "total_dynamics_samples": len(data["dynamics_samples"]),
            "total_harmony_samples": len(data["harmony_samples"]),
            "total_groove_samples": len(data["groove_samples"]),
        }

        return data

    def _build_emotion_dynamics_map(self) -> List[Dict[str, Any]]:
        """Build emotion-to-dynamics mapping samples for training."""
        emotions = [
            "melancholy", "sad", "grief", "angry", "anxious",
            "happy", "joyful", "excited", "peaceful", "content",
            "nostalgic", "hopeful",
        ]
        sections = ["intro", "verse", "chorus", "bridge", "outro"]

        samples = []
        for emotion in emotions:
            for section in sections:
                dynamics = self.get_dynamics_for_emotion(emotion, section)
                samples.append({
                    "emotion": emotion,
                    "section": section,
                    "dynamics": dynamics,
                })

        return samples

    def record_harmony_analysis(self, analysis: Dict[str, Any]):
        """Record harmony analysis for training data export."""
        if not hasattr(self, "_harmony_history"):
            self._harmony_history = []
        self._harmony_history.append({
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "analysis": analysis,
        })

    def record_groove_analysis(self, analysis: Dict[str, Any]):
        """Record groove analysis for training data export."""
        if not hasattr(self, "_groove_history"):
            self._groove_history = []
        self._groove_history.append({
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "analysis": analysis,
        })

    # =========================================================================
    # Unified Processing
    # =========================================================================

    def process_complete(
        self,
        audio: Optional[Any] = None,
        midi_notes: Optional[List[tuple]] = None,
        section: Optional[str] = None,
        emotion: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run complete integrated processing.

        Args:
            audio: Optional audio buffer
            midi_notes: Optional MIDI notes
            section: Current section
            emotion: Current emotion

        Returns:
            Complete analysis results
        """
        result = {}

        # Harmony analysis
        if midi_notes:
            result["harmony"] = self.process_notes(midi_notes)

        # Dynamics
        if emotion:
            result["dynamics"] = self.get_dynamics_for_emotion(emotion, section or "verse")

        # Add suggestions
        if "harmony" in result:
            result["chord_suggestions"] = self.get_chord_suggestions(
                result["harmony"].get("chord", {}),
                emotion
            )

        return result
