"""
Interrogator Engine - Therapist-style Q&A that builds musical intent.

Part of Kelly MIDI Companion Tier 1 MVP.

Philosophy: "Interrogate Before Generate". We don't ask "what tempo?"
We ask "does this feel urgent or does time feel frozen?" The questions
reveal the emotional truth. The answers become musical parameters.
Kelly understands emotions like a therapist but NEVER voices psychological
observations - it translates silently into music.

Three-Phase Intent Building:
1. WOUND - What hurts? What's the core experience?
2. EMOTION - How does it feel? What's the texture?
3. RULE-BREAKS - What conventions should we violate for authenticity?

Integration:
    from kellymidicompanion_interrogator import Interrogator, InterrogationSession

    interrogator = Interrogator()
    session = interrogator.start_session()

    # Get first question
    q = session.get_next_question()
    print(q.text)

    # User responds
    session.answer(q.id, "it feels like drowning")

    # Continue until intent is built
    while not session.is_complete():
        q = session.get_next_question()
        session.answer(q.id, user_input)

    # Get final intent
    intent = session.build_intent()
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import random


# =============================================================================
# ENUMS
# =============================================================================

class InterrogationPhase(Enum):
    """The three phases of intent building."""
    WOUND = "wound"         # Core experience/desire
    EMOTION = "emotion"     # Feeling texture
    RULE_BREAK = "rule_break"  # What to violate


class QuestionType(Enum):
    """Types of questions."""
    OPEN = "open"               # Free text response
    CHOICE = "choice"           # Select from options
    SCALE = "scale"             # 1-10 rating
    BINARY = "binary"           # Yes/No
    METAPHOR = "metaphor"       # Comparative/imagery


class EmotionCategory(Enum):
    """Primary emotion categories (Plutchik)."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"


# =============================================================================
# QUESTION DATABASE
# =============================================================================

WOUND_QUESTIONS = [
    {
        "id": "wound_core",
        "text": "What's the feeling you're trying to express? Not the story, but the raw sensation.",  # noqa: E501

        "type": QuestionType.OPEN,
        "phase": InterrogationPhase.WOUND,
        "weight": 1.0,
        "follow_ups": ["wound_physical", "wound_metaphor"],
    },
    {
        "id": "wound_physical",
        "text": "Where do you feel it in your body?",
        "type": QuestionType.OPEN,
        "phase": InterrogationPhase.WOUND,
        "weight": 0.8,
        "maps_to": "intensity",
    },
    {
        "id": "wound_metaphor",
        "text": "If this feeling were weather, what would it be?",
        "type": QuestionType.METAPHOR,
        "phase": InterrogationPhase.WOUND,
        "weight": 0.7,
        "choices": [
            ("storm", {"arousal": 0.9, "valence": -0.6}),
            ("fog", {"arousal": 0.2, "valence": -0.3}),
            ("rain", {"arousal": 0.4, "valence": -0.4}),
            ("sunshine", {"arousal": 0.7, "valence": 0.8}),
            ("wind", {"arousal": 0.6, "valence": 0.0}),
            ("snow", {"arousal": 0.3, "valence": 0.1}),
            ("drought", {"arousal": 0.2, "valence": -0.5}),
            ("hurricane", {"arousal": 1.0, "valence": -0.8}),
        ],
    },
    {
        "id": "wound_time",
        "text": "Does this feel like something that just happened, or something you've carried for years?",  # noqa: E501

        "type": QuestionType.CHOICE,
        "phase": InterrogationPhase.WOUND,
        "weight": 0.6,
        "choices": [
            ("just happened", {"temporal": "acute"}),
            ("recent weeks", {"temporal": "recent"}),
            ("months", {"temporal": "chronic"}),
            ("years", {"temporal": "deep"}),
            ("always been there", {"temporal": "foundational"}),
        ],
    },
    {
        "id": "wound_direction",
        "text": "Is this something you want to move through, or something you need to sit with?",
        "type": QuestionType.BINARY,
        "phase": InterrogationPhase.WOUND,
        "weight": 0.7,
        "choices": [
            ("move through", {"narrative_arc": "transformation"}),
            ("sit with", {"narrative_arc": "dwelling"}),
        ],
    },
    {
        "id": "wound_witness",
        "text": "Is this for you alone, or do you want someone else to understand?",
        "type": QuestionType.BINARY,
        "phase": InterrogationPhase.WOUND,
        "weight": 0.5,
        "choices": [
            ("for me", {"audience": "self"}),
            ("to be understood", {"audience": "other"}),
        ],
    },
]

EMOTION_QUESTIONS = [
    {
        "id": "emotion_primary",
        "text": "Which of these feels closest to the center of it?",
        "type": QuestionType.CHOICE,
        "phase": InterrogationPhase.EMOTION,
        "weight": 1.0,
        "choices": [
            ("grief - the weight of loss", {"primary_emotion": "grief", "valence": -0.8, "arousal": 0.3}),  # noqa: E501

            ("anger - the heat of injustice", {"primary_emotion": "anger", "valence": -0.6, "arousal": 0.9}),  # noqa: E501

            ("fear - the cold of uncertainty", {"primary_emotion": "fear", "valence": -0.7, "arousal": 0.8}),  # noqa: E501

            ("longing - the ache of distance", {"primary_emotion": "longing", "valence": -0.4, "arousal": 0.4}),  # noqa: E501

            ("hope - the warmth of possibility", {"primary_emotion": "hope", "valence": 0.6, "arousal": 0.5}),  # noqa: E501

            ("joy - the lightness of being", {"primary_emotion": "joy", "valence": 0.8, "arousal": 0.7}),  # noqa: E501

            ("love - the fullness of connection", {"primary_emotion": "love", "valence": 0.7, "arousal": 0.6}),  # noqa: E501

            ("numbness - the absence of feeling", {"primary_emotion": "dissociation", "valence": 0.0, "arousal": 0.1}),  # noqa: E501

        ],
    },
    {
        "id": "emotion_intensity",
        "text": "On a scale of 1-10, how intense is this feeling right now?",
        "type": QuestionType.SCALE,
        "phase": InterrogationPhase.EMOTION,
        "weight": 0.9,
        "maps_to": "intensity",
    },
    {
        "id": "emotion_texture",
        "text": "What's the texture of this feeling?",
        "type": QuestionType.CHOICE,
        "phase": InterrogationPhase.EMOTION,
        "weight": 0.8,
        "choices": [
            ("sharp and jagged", {"texture": "sharp", "attack": "hard"}),
            ("dull and heavy", {"texture": "dull", "attack": "soft"}),
            ("waves that come and go", {"texture": "wave", "rhythm": "varying"}),
            ("constant pressure", {"texture": "sustained", "rhythm": "steady"}),
            ("scattered fragments", {"texture": "fragmented", "rhythm": "irregular"}),
            ("hollow emptiness", {"texture": "hollow", "density": "sparse"}),
        ],
    },
    {
        "id": "emotion_movement",
        "text": "Does this feeling want to move fast or slow?",
        "type": QuestionType.SCALE,
        "phase": InterrogationPhase.EMOTION,
        "weight": 0.7,
        "maps_to": "tempo_feel",
        "scale_labels": ("frozen still", "crawling", "walking", "running", "racing"),
    },
    {
        "id": "emotion_color",
        "text": "If this feeling had a color, what would it be?",
        "type": QuestionType.CHOICE,
        "phase": InterrogationPhase.EMOTION,
        "weight": 0.5,
        "choices": [
            ("black", {"brightness": 0.1, "mode_tendency": "minor"}),
            ("deep blue", {"brightness": 0.3, "mode_tendency": "minor"}),
            ("grey", {"brightness": 0.4, "mode_tendency": "neutral"}),
            ("red", {"brightness": 0.6, "mode_tendency": "dissonant"}),
            ("gold", {"brightness": 0.8, "mode_tendency": "major"}),
            ("white", {"brightness": 0.9, "mode_tendency": "major"}),
            ("purple", {"brightness": 0.5, "mode_tendency": "modal"}),
        ],
    },
    {
        "id": "emotion_secondary",
        "text": "What else is mixed in with this feeling?",
        "type": QuestionType.CHOICE,
        "phase": InterrogationPhase.EMOTION,
        "weight": 0.6,
        "multi_select": True,
        "choices": [
            ("guilt", {"secondary": "guilt"}),
            ("shame", {"secondary": "shame"}),
            ("regret", {"secondary": "regret"}),
            ("tenderness", {"secondary": "tenderness"}),
            ("defiance", {"secondary": "defiance"}),
            ("acceptance", {"secondary": "acceptance"}),
            ("nothing else", {}),
        ],
    },
    {
        "id": "emotion_breath",
        "text": "How does this feeling affect your breathing?",
        "type": QuestionType.CHOICE,
        "phase": InterrogationPhase.EMOTION,
        "weight": 0.5,
        "choices": [
            ("can't catch my breath", {"phrase_length": "short", "rest_density": "low"}),
            ("holding my breath", {"phrase_length": "long", "rest_density": "high"}),
            ("sighing", {"phrase_length": "medium", "rest_density": "medium"}),
            ("breathing normally", {"phrase_length": "regular", "rest_density": "normal"}),
            ("deep heavy breaths", {"phrase_length": "long", "rest_density": "low"}),
        ],
    },
]

RULE_BREAK_QUESTIONS = [
    {
        "id": "rule_resolution",
        "text": "Should this music resolve and find peace, or stay unresolved and restless?",
        "type": QuestionType.BINARY,
        "phase": InterrogationPhase.RULE_BREAK,
        "weight": 1.0,
        "choices": [
            ("find resolution", {"resolve": True, "rule_break": None}),
            ("stay unresolved", {"resolve": False, "rule_break": "avoid_resolution"}),
        ],
    },
    {
        "id": "rule_perfection",
        "text": "Should this sound polished and perfect, or raw and imperfect?",
        "type": QuestionType.SCALE,
        "phase": InterrogationPhase.RULE_BREAK,
        "weight": 0.9,
        "maps_to": "humanization",
        "scale_labels": ("pristine", "clean", "natural", "rough", "broken"),
    },
    {
        "id": "rule_convention",
        "text": "Are there any musical 'rules' you want to intentionally break?",
        "type": QuestionType.CHOICE,
        "phase": InterrogationPhase.RULE_BREAK,
        "weight": 0.8,
        "multi_select": True,
        "choices": [
            ("wrong notes that feel right", {"rule_break": "chromatic_tension"}),
            ("rhythms that stumble", {"rule_break": "rhythmic_displacement"}),
            ("chords that 'shouldn't' work", {"rule_break": "non_functional_harmony"}),
            ("melodies that don't resolve", {"rule_break": "melodic_avoidance"}),
            ("uncomfortable silences", {"rule_break": "extended_rests"}),
            ("none - keep it conventional", {}),
        ],
    },
    {
        "id": "rule_beauty",
        "text": "Does this need to be beautiful, or is ugly more honest?",
        "type": QuestionType.SCALE,
        "phase": InterrogationPhase.RULE_BREAK,
        "weight": 0.7,
        "maps_to": "dissonance_tolerance",
        "scale_labels": ("beautiful", "bittersweet", "neutral", "uncomfortable", "harsh"),
    },
    {
        "id": "rule_predictable",
        "text": "Should listeners know what's coming next, or be kept off-balance?",
        "type": QuestionType.SCALE,
        "phase": InterrogationPhase.RULE_BREAK,
        "weight": 0.6,
        "maps_to": "predictability",
        "scale_labels": ("predictable", "mostly expected", "some surprises", "unpredictable", "chaotic"),  # noqa: E501

    },
    {
        "id": "rule_genre",
        "text": "How closely should this follow genre conventions?",
        "type": QuestionType.CHOICE,
        "phase": InterrogationPhase.RULE_BREAK,
        "weight": 0.5,
        "choices": [
            ("strictly follow the genre", {"genre_adherence": 1.0}),
            ("mostly follow with some variation", {"genre_adherence": 0.7}),
            ("use genre as starting point only", {"genre_adherence": 0.4}),
            ("ignore genre entirely", {"genre_adherence": 0.0}),
        ],
    },
]


# =============================================================================
# KEYWORD ANALYSIS
# =============================================================================

KEYWORD_EMOTION_MAP = {
    # Grief family
    "loss": {"emotion": "grief", "valence": -0.8, "arousal": 0.3},
    "gone": {"emotion": "grief", "valence": -0.7, "arousal": 0.3},
    "miss": {"emotion": "longing", "valence": -0.5, "arousal": 0.4},
    "died": {"emotion": "grief", "valence": -0.9, "arousal": 0.4},
    "death": {"emotion": "grief", "valence": -0.9, "arousal": 0.4},
    "funeral": {"emotion": "grief", "valence": -0.8, "arousal": 0.3},
    "empty": {"emotion": "grief", "valence": -0.6, "arousal": 0.2},
    "hollow": {"emotion": "dissociation", "valence": -0.4, "arousal": 0.1},

    # Anger family
    "angry": {"emotion": "anger", "valence": -0.6, "arousal": 0.9},
    "furious": {"emotion": "rage", "valence": -0.8, "arousal": 1.0},
    "hate": {"emotion": "rage", "valence": -0.9, "arousal": 0.8},
    "betrayed": {"emotion": "anger", "valence": -0.7, "arousal": 0.7},
    "unfair": {"emotion": "defiance", "valence": -0.5, "arousal": 0.7},

    # Fear family
    "scared": {"emotion": "fear", "valence": -0.7, "arousal": 0.8},
    "terrified": {"emotion": "fear", "valence": -0.9, "arousal": 0.9},
    "anxious": {"emotion": "anxiety", "valence": -0.5, "arousal": 0.7},
    "worried": {"emotion": "anxiety", "valence": -0.4, "arousal": 0.6},
    "panic": {"emotion": "fear", "valence": -0.8, "arousal": 1.0},
    "dread": {"emotion": "fear", "valence": -0.7, "arousal": 0.6},

    # Joy family
    "happy": {"emotion": "joy", "valence": 0.7, "arousal": 0.6},
    "excited": {"emotion": "euphoria", "valence": 0.8, "arousal": 0.9},
    "peaceful": {"emotion": "tenderness", "valence": 0.5, "arousal": 0.2},
    "grateful": {"emotion": "joy", "valence": 0.6, "arousal": 0.4},
    "love": {"emotion": "love", "valence": 0.7, "arousal": 0.6},

    # Complex
    "numb": {"emotion": "dissociation", "valence": 0.0, "arousal": 0.1},
    "confused": {"emotion": "anxiety", "valence": -0.3, "arousal": 0.5},
    "overwhelmed": {"emotion": "anxiety", "valence": -0.5, "arousal": 0.8},
    "stuck": {"emotion": "melancholy", "valence": -0.4, "arousal": 0.2},
    "trapped": {"emotion": "anxiety", "valence": -0.6, "arousal": 0.7},
    "drowning": {"emotion": "grief", "valence": -0.7, "arousal": 0.6},
    "floating": {"emotion": "dissociation", "valence": 0.0, "arousal": 0.2},
    "burning": {"emotion": "rage", "valence": -0.6, "arousal": 0.9},
    "frozen": {"emotion": "fear", "valence": -0.5, "arousal": 0.2},
    "heavy": {"emotion": "grief", "valence": -0.5, "arousal": 0.2},
    "light": {"emotion": "hope", "valence": 0.5, "arousal": 0.5},
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Question:
    """A single interrogation question."""
    id: str
    text: str
    type: QuestionType
    phase: InterrogationPhase
    weight: float = 1.0
    choices: Optional[List[Tuple[str, Dict]]] = None
    maps_to: Optional[str] = None
    scale_labels: Optional[Tuple] = None
    multi_select: bool = False
    follow_ups: List[str] = field(default_factory=list)


@dataclass
class Answer:
    """User's answer to a question."""
    question_id: str
    raw_value: Any           # The actual response
    interpreted: Dict        # Parsed musical parameters
    confidence: float = 1.0  # How confident in interpretation


@dataclass
class MusicalIntent:
    """The final intent built from interrogation."""
    # Core emotional parameters
    primary_emotion: str = "neutral"
    secondary_emotions: List[str] = field(default_factory=list)
    valence: float = 0.0           # -1 to 1
    arousal: float = 0.5           # 0 to 1
    intensity: float = 0.5         # 0 to 1

    # Temporal
    temporal_quality: str = "present"  # acute, recent, chronic, deep
    narrative_arc: str = "linear"      # transformation, dwelling, release

    # Musical directives
    mode_tendency: str = "minor"       # major, minor, modal, dissonant
    tempo_feel: str = "moderate"       # frozen, slow, moderate, fast, frantic
    texture: str = "sustained"         # sharp, dull, wave, sustained, fragmented
    density: str = "moderate"          # sparse, moderate, dense

    # Rule breaks
    rule_breaks: List[str] = field(default_factory=list)
    resolve: bool = True
    humanization: float = 0.5          # 0 = perfect, 1 = broken
    dissonance_tolerance: float = 0.3  # 0 = consonant, 1 = harsh
    genre_adherence: float = 0.5

    # Metadata
    audience: str = "self"
    confidence_score: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export."""
        return {
            "primary_emotion": self.primary_emotion,
            "secondary_emotions": self.secondary_emotions,
            "valence": self.valence,
            "arousal": self.arousal,
            "intensity": self.intensity,
            "temporal_quality": self.temporal_quality,
            "narrative_arc": self.narrative_arc,
            "mode_tendency": self.mode_tendency,
            "tempo_feel": self.tempo_feel,
            "texture": self.texture,
            "density": self.density,
            "rule_breaks": self.rule_breaks,
            "resolve": self.resolve,
            "humanization": self.humanization,
            "dissonance_tolerance": self.dissonance_tolerance,
            "genre_adherence": self.genre_adherence,
            "audience": self.audience,
            "confidence_score": self.confidence_score,
        }

    def get_suggested_tempo(self) -> int:
        """Convert tempo_feel to BPM suggestion."""
        tempo_map = {
            "frozen": 50,
            "slow": 70,
            "moderate": 95,
            "fast": 130,
            "frantic": 170,
        }
        return tempo_map.get(self.tempo_feel, 100)

    def get_suggested_key(self) -> str:
        """Suggest a key based on emotion."""
        emotion_keys = {
            "grief": "F",
            "anger": "E",
            "fear": "B",
            "longing": "A",
            "hope": "G",
            "joy": "D",
            "love": "C",
            "dissociation": "F#",
            "anxiety": "C#",
            "rage": "Eb",
            "tenderness": "Ab",
            "defiance": "E",
            "melancholy": "Dm",
            "euphoria": "A",
            "nostalgia": "G",
        }
        return emotion_keys.get(self.primary_emotion, "C")


@dataclass
class InterrogationState:
    """Current state of an interrogation session."""
    phase: InterrogationPhase = InterrogationPhase.WOUND
    questions_asked: List[str] = field(default_factory=list)
    answers: List[Answer] = field(default_factory=list)
    accumulated_params: Dict = field(default_factory=dict)
    is_complete: bool = False
    min_questions_per_phase: int = 2
    max_questions_per_phase: int = 4


# =============================================================================
# INTERROGATION SESSION
# =============================================================================

class InterrogationSession:
    """
    A single interrogation session that builds intent through Q&A.

    Usage:
        session = InterrogationSession()

        while not session.is_complete():
            q = session.get_next_question()
            print(q.text)
            user_response = input()
            session.answer(q.id, user_response)

        intent = session.build_intent()
    """

    def __init__(
        self,
        min_questions: int = 6,
        max_questions: int = 12,
        skip_phases: Optional[List[InterrogationPhase]] = None
    ):
        self.state = InterrogationState()
        self.min_questions = min_questions
        self.max_questions = max_questions
        self.skip_phases = skip_phases or []

        # Build question pool
        self.questions = {}
        for q_data in WOUND_QUESTIONS + EMOTION_QUESTIONS + RULE_BREAK_QUESTIONS:
            q = Question(
                id=q_data["id"],
                text=q_data["text"],
                type=q_data["type"],
                phase=q_data["phase"],
                weight=q_data.get("weight", 1.0),
                choices=q_data.get("choices"),
                maps_to=q_data.get("maps_to"),
                scale_labels=q_data.get("scale_labels"),
                multi_select=q_data.get("multi_select", False),
                follow_ups=q_data.get("follow_ups", []),
            )
            self.questions[q.id] = q

        self._pending_follow_ups = []

    def get_next_question(self) -> Optional[Question]:
        """Get the next question to ask."""
        if self.state.is_complete:
            return None

        # Check for pending follow-ups
        if self._pending_follow_ups:
            follow_id = self._pending_follow_ups.pop(0)
            if follow_id in self.questions and follow_id not in self.state.questions_asked:
                return self.questions[follow_id]

        # Get questions for current phase
        phase_questions = [
            q for q in self.questions.values()
            if q.phase == self.state.phase
            and q.id not in self.state.questions_asked
        ]

        if not phase_questions:
            # Move to next phase
            self._advance_phase()
            return self.get_next_question()

        # Check if we've asked enough for this phase
        phase_asked = len([
            a for a in self.state.answers
            if self.questions[a.question_id].phase == self.state.phase
        ])

        if phase_asked >= self.state.max_questions_per_phase:
            self._advance_phase()
            return self.get_next_question()

        # Weight by importance and randomize slightly
        phase_questions.sort(key=lambda q: -q.weight + random.uniform(0, 0.3))

        return phase_questions[0]

    def answer(self, question_id: str, response: Any) -> Dict:
        """
        Record an answer and return interpreted parameters.

        Args:
            question_id: ID of the question being answered
            response: User's response (text, number, or choice index)

        Returns:
            Dictionary of interpreted musical parameters
        """
        if question_id not in self.questions:
            return {}

        question = self.questions[question_id]
        interpreted = self._interpret_answer(question, response)

        answer = Answer(
            question_id=question_id,
            raw_value=response,
            interpreted=interpreted,
        )

        self.state.answers.append(answer)
        self.state.questions_asked.append(question_id)

        # Merge into accumulated params
        for key, value in interpreted.items():
            if key in self.state.accumulated_params:
                # Average numeric values
                if isinstance(
                        value, (int, float)) and isinstance(
                        self.state.accumulated_params[key],
                        (int, float)):
                    self.state.accumulated_params[key] = (
                        self.state.accumulated_params[key] + value) / 2
                elif isinstance(value, list):
                    self.state.accumulated_params[key].extend(value)
                else:
                    self.state.accumulated_params[key] = value
            else:
                self.state.accumulated_params[key] = value

        # Add follow-ups
        if question.follow_ups:
            self._pending_follow_ups.extend(question.follow_ups)

        # Check completion
        self._check_completion()

        return interpreted

    def _interpret_answer(self, question: Question, response: Any) -> Dict:
        """Interpret a response into musical parameters."""
        params = {}

        if question.type == QuestionType.OPEN:
            # Analyze free text for keywords
            params = self._analyze_text(str(response))

        elif question.type == QuestionType.CHOICE:
            # Direct mapping from choice
            if question.choices and isinstance(response, (int, str)):
                if isinstance(response, int) and 0 <= response < len(question.choices):
                    _, choice_params = question.choices[response]
                    params = choice_params.copy()
                elif isinstance(response, str):
                    # Match by text
                    response_lower = response.lower()
                    for text, choice_params in question.choices:
                        if response_lower in text.lower() or text.lower() in response_lower:
                            params = choice_params.copy()
                            break

        elif question.type == QuestionType.SCALE:
            # Convert 1-10 to 0-1 range
            try:
                value = float(response)
                normalized = (value - 1) / 9  # 1-10 -> 0-1
                if question.maps_to:
                    params[question.maps_to] = normalized

                    # Also map to descriptive label
                    if question.scale_labels:
                        label_idx = min(
                            int(normalized * len(question.scale_labels)),
                            len(question.scale_labels) - 1)
                        params[f"{question.maps_to}_label"] = question.scale_labels[label_idx]
            except (ValueError, TypeError):
                pass

        elif question.type == QuestionType.BINARY:
            # Yes/No or A/B
            if question.choices and len(question.choices) == 2:
                response_str = str(response).lower()
                if response_str in ["yes", "y", "1", "true", "a"] or response_str in question.choices[0][0].lower():  # noqa: E501

                    params = question.choices[0][1].copy()
                else:
                    params = question.choices[1][1].copy()

        elif question.type == QuestionType.METAPHOR:
            # Similar to choice but with richer mapping
            if question.choices:
                response_lower = str(response).lower()
                for text, choice_params in question.choices:
                    if text.lower() in response_lower or response_lower in text.lower():
                        params = choice_params.copy()
                        break

        return params

    def _analyze_text(self, text: str) -> Dict:
        """Analyze free text for emotional keywords."""
        text_lower = text.lower()
        params = {
            "valence": 0.0,
            "arousal": 0.5,
            "detected_keywords": [],
        }

        matches = 0
        for keyword, mapping in KEYWORD_EMOTION_MAP.items():
            if keyword in text_lower:
                params["detected_keywords"].append(keyword)

                if "emotion" in mapping:
                    params["detected_emotion"] = mapping["emotion"]
                if "valence" in mapping:
                    params["valence"] += mapping["valence"]
                    matches += 1
                if "arousal" in mapping:
                    params["arousal"] += mapping["arousal"]

        # Average the accumulated values
        if matches > 0:
            params["valence"] /= matches
            params["arousal"] /= matches

        return params

    def _advance_phase(self):
        """Move to the next interrogation phase."""
        phases = [InterrogationPhase.WOUND,
                  InterrogationPhase.EMOTION, InterrogationPhase.RULE_BREAK]

        try:
            current_idx = phases.index(self.state.phase)
            next_idx = current_idx + 1

            while next_idx < len(phases):
                if phases[next_idx] not in self.skip_phases:
                    self.state.phase = phases[next_idx]
                    return
                next_idx += 1

            # No more phases
            self.state.is_complete = True
        except ValueError:
            self.state.is_complete = True

    def _check_completion(self):
        """Check if interrogation is complete."""
        total_questions = len(self.state.answers)

        if total_questions >= self.max_questions:
            self.state.is_complete = True
        elif total_questions >= self.min_questions and self.state.phase == InterrogationPhase.RULE_BREAK:  # noqa: E501

            # Can complete early if we have enough
            phase_counts = {phase: 0 for phase in InterrogationPhase}
            for answer in self.state.answers:
                q = self.questions[answer.question_id]
                phase_counts[q.phase] += 1

            if all(count >= 1 for phase, count in phase_counts.items()
                   if phase not in self.skip_phases):
                self.state.is_complete = True

    def is_complete(self) -> bool:
        """Check if session is complete."""
        return self.state.is_complete

    def build_intent(self) -> MusicalIntent:
        """Build final MusicalIntent from all answers."""
        params = self.state.accumulated_params

        intent = MusicalIntent(
            primary_emotion=params.get(
                "primary_emotion", params.get("detected_emotion", "neutral")),
            secondary_emotions=[params.get("secondary")] if params.get("secondary") else [],
            valence=params.get("valence", 0.0),
            arousal=params.get("arousal", 0.5),
            intensity=params.get("intensity", 0.5),
            temporal_quality=params.get("temporal", "present"),
            narrative_arc=params.get("narrative_arc", "linear"),
            mode_tendency=params.get(
                "mode_tendency", "minor" if params.get("valence", 0) < 0 else "major"),
            tempo_feel=params.get("tempo_feel_label", "moderate"),
            texture=params.get("texture", "sustained"),
            density="sparse" if params.get("rest_density") == "high" else "moderate",
            rule_breaks=[params.get("rule_break")] if params.get("rule_break") else [],
            resolve=params.get("resolve", True),
            humanization=params.get("humanization", 0.5),
            dissonance_tolerance=params.get("dissonance_tolerance", 0.3),
            genre_adherence=params.get("genre_adherence", 0.5),
            audience=params.get("audience", "self"),
            confidence_score=len(self.state.answers) / self.max_questions,)

        return intent

    def get_summary(self) -> str:
        """Get a summary of the session so far."""
        lines = [
            f"Phase: {self.state.phase.value}",
            f"Questions asked: {len(self.state.answers)}",
            f"Complete: {self.state.is_complete}",
            "",
            "Accumulated parameters:",
        ]

        for key, value in self.state.accumulated_params.items():
            lines.append(f"  {key}: {value}")

        return "\n".join(lines)


# =============================================================================
# MAIN INTERROGATOR
# =============================================================================

class Interrogator:
    """
    Main interrogator class for creating and managing sessions.

    Usage:
        interrogator = Interrogator()
        session = interrogator.start_session()

        # Or quick interrogation
        intent = interrogator.quick_interrogate("I feel like I'm drowning in grief")
    """

    def __init__(self):
        pass

    def start_session(
        self,
        min_questions: int = 6,
        max_questions: int = 12,
        skip_phases: Optional[List[InterrogationPhase]] = None
    ) -> InterrogationSession:
        """Start a new interrogation session."""
        return InterrogationSession(
            min_questions=min_questions,
            max_questions=max_questions,
            skip_phases=skip_phases,
        )

    def quick_interrogate(self, description: str) -> MusicalIntent:
        """
        Quick interrogation from a single text description.
        Analyzes keywords to build intent without full Q&A.

        Args:
            description: Free text describing the feeling

        Returns:
            MusicalIntent based on text analysis
        """
        session = InterrogationSession(min_questions=1, max_questions=1)

        # Use the wound_core question with the description
        session.answer("wound_core", description)

        # Add some defaults based on analysis
        params = session.state.accumulated_params

        intent = MusicalIntent(
            primary_emotion=params.get("detected_emotion", "neutral"),
            valence=params.get("valence", 0.0),
            arousal=params.get("arousal", 0.5),
            intensity=abs(params.get("valence", 0.0)),
            mode_tendency="minor" if params.get("valence", 0) < 0 else "major",
            humanization=0.3,
            confidence_score=0.3,  # Low confidence for quick interrogation
        )

        return intent

    def from_emotion(self, emotion: str, intensity: float = 0.7) -> MusicalIntent:
        """
        Create intent directly from emotion name.
        Bypasses interrogation for when emotion is known.

        Args:
            emotion: Emotion name (grief, hope, rage, etc.)
            intensity: 0-1 intensity

        Returns:
            MusicalIntent configured for that emotion
        """
        emotion_presets = {
            "grief": {"valence": -0.8, "arousal": 0.3, "mode": "minor", "tempo": "slow"},
            "hope": {"valence": 0.6, "arousal": 0.5, "mode": "major", "tempo": "moderate"},
            "rage": {"valence": -0.6, "arousal": 0.9, "mode": "dissonant", "tempo": "fast"},
            "fear": {"valence": -0.7, "arousal": 0.8, "mode": "minor", "tempo": "moderate"},
            "joy": {"valence": 0.8, "arousal": 0.7, "mode": "major", "tempo": "fast"},
            "longing": {"valence": -0.4, "arousal": 0.4, "mode": "minor", "tempo": "slow"},
            "anxiety": {"valence": -0.5, "arousal": 0.7, "mode": "dissonant", "tempo": "moderate"},
            "tenderness": {"valence": 0.5, "arousal": 0.2, "mode": "major", "tempo": "slow"},
            "defiance": {"valence": -0.3, "arousal": 0.8, "mode": "minor", "tempo": "fast"},
            "melancholy": {"valence": -0.5, "arousal": 0.2, "mode": "minor", "tempo": "slow"},
            "nostalgia": {"valence": 0.2, "arousal": 0.4, "mode": "major", "tempo": "moderate"},
            "euphoria": {"valence": 0.9, "arousal": 0.9, "mode": "major", "tempo": "fast"},
            "dissociation": {"valence": 0.0, "arousal": 0.1, "mode": "modal", "tempo": "slow"},
        }

        preset = emotion_presets.get(emotion.lower(), {
            "valence": 0.0, "arousal": 0.5, "mode": "minor", "tempo": "moderate"
        })

        return MusicalIntent(
            primary_emotion=emotion.lower(),
            valence=preset["valence"] * intensity,
            arousal=preset["arousal"],
            intensity=intensity,
            mode_tendency=preset["mode"],
            tempo_feel=preset["tempo"],
            confidence_score=0.8,
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def interrogate_for_grief() -> MusicalIntent:
    """Quick grief intent."""
    return Interrogator().from_emotion("grief", 0.8)


def interrogate_from_text(text: str) -> MusicalIntent:
    """Quick intent from description."""
    return Interrogator().quick_interrogate(text)


# =============================================================================
# EXAMPLE
# =============================================================================

if __name__ == "__main__":
    print("=== INTERROGATOR ENGINE DEMO ===\n")

    # Quick interrogation
    print("--- Quick Interrogation ---")
    intent = Interrogator().quick_interrogate(
        "I feel like I'm drowning in grief, everything feels heavy and slow"
    )
    print(f"Detected emotion: {intent.primary_emotion}")
    print(f"Valence: {intent.valence:.2f}")
    print(f"Arousal: {intent.arousal:.2f}")
    print(f"Suggested key: {intent.get_suggested_key()}")
    print(f"Suggested tempo: {intent.get_suggested_tempo()} BPM")
    print()

    # From emotion
    print("--- From Emotion ---")
    for emotion in ["grief", "hope", "rage"]:
        intent = Interrogator().from_emotion(emotion)
        print(f"{emotion}: key={intent.get_suggested_key()}, tempo={intent.get_suggested_tempo()}, mode={intent.mode_tendency}")  # noqa: E501

    print()

    # Interactive demo
    print("--- Interactive Session (simulated) ---")
    session = Interrogator().start_session(min_questions=3, max_questions=5)

    # Simulate answers
    simulated_answers = [
        ("wound_core", "I lost someone and I can't stop thinking about them"),
        ("emotion_primary", 0),  # grief
        ("emotion_intensity", 8),
        ("rule_resolution", "stay unresolved"),
    ]

    for q_id, answer in simulated_answers:
        if session.is_complete():
            break
        q = session.get_next_question()
        if q:
            print(f"Q: {q.text}")
            # Use our simulated answer if it matches, otherwise skip
            if q.id == q_id:
                print(f"A: {answer}")
                session.answer(q.id, answer)
            else:
                print(f"(skipped - expected {q_id})")

    final_intent = session.build_intent()
    print("\nFinal Intent:")
    print(f"  Emotion: {final_intent.primary_emotion}")
    print(f"  Resolve: {final_intent.resolve}")
    print(f"  Confidence: {final_intent.confidence_score:.2f}")
