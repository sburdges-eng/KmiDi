"""
Learning Module - AI-powered instrument education with web-sourced curricula.

This module enables AI to:
- Fetch training programs from various educational sites
- Organize content by instrument and difficulty level
- Generate personalized learning paths
- Adapt teaching methodology to student needs

Philosophy: "Meet the student where they are, take them where they need to go."
"""

from music_brain.learning.curriculum import (
    Curriculum,
    CurriculumBuilder,
    DifficultyLevel,
    LearningObjective,
    LearningPath,
    Lesson,
    Module,
    SkillCategory,
)
from music_brain.learning.instruments import (
    INSTRUMENTS,
    Instrument,
    InstrumentFamily,
    get_beginner_instruments,
    get_instrument,
    get_instruments_by_family,
)
from music_brain.learning.openweight_learning import (
    OpenWeightLearner,
    OpenWeightLearningManager,
)
from music_brain.learning.pedagogy import (
    AdaptiveTeacher,
    LearningPreference,
    PedagogyEngine,
    StudentProfile,
    TeachingStyle,
    generate_ai_teaching_prompt,
)
from music_brain.learning.resources import (
    KNOWN_SOURCES,
    LearningResource,
    ResourceCache,
    ResourceFetcher,
    ResourceType,
    generate_learning_plan,
    get_recommended_sources,
)

__all__ = [
    # Curriculum
    "DifficultyLevel",
    "SkillCategory",
    "LearningObjective",
    "Lesson",
    "Module",
    "Curriculum",
    "LearningPath",
    "CurriculumBuilder",
    # Resources
    "ResourceType",
    "LearningResource",
    "ResourceFetcher",
    "ResourceCache",
    "KNOWN_SOURCES",
    "get_recommended_sources",
    "generate_learning_plan",
    # Instruments
    "InstrumentFamily",
    "Instrument",
    "INSTRUMENTS",
    "get_instrument",
    "get_instruments_by_family",
    "get_beginner_instruments",
    # Pedagogy
    "TeachingStyle",
    "StudentProfile",
    "LearningPreference",
    "AdaptiveTeacher",
    "PedagogyEngine",
    "generate_ai_teaching_prompt",
    # OpenWeight Learning
    "OpenWeightLearner",
    "OpenWeightLearningManager",
]
