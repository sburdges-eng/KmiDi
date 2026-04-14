"""HTTP/API Pydantic models aligned with docs/prd/KMIDI_PRD.md (TTG v1, versioning)."""

from music_brain.api_schemas.generate_v1 import (
    EmotionalIntent,
    GenerateRequest,
    InterrogateRequest,
    LyricsRequest,
    TechnicalIntent,
)
from music_brain.api_schemas.ttg_v1 import (
    EnergyCurvePointV1,
    EnergyCurveV1,
    OrchestrationRoleSpecV1,
    TTGOrchestrationV1,
    TTGPhraseV1,
    TTGMovementV1,
)

__all__ = [
    "EmotionalIntent",
    "EnergyCurvePointV1",
    "EnergyCurveV1",
    "GenerateRequest",
    "InterrogateRequest",
    "LyricsRequest",
    "OrchestrationRoleSpecV1",
    "TechnicalIntent",
    "TTGOrchestrationV1",
    "TTGMovementV1",
    "TTGPhraseV1",
]
