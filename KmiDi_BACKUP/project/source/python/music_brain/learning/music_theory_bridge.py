"""
Music Theory Bridge - Python interface for C++ MusicTheoryBridge

This module provides bridge functions that are called from C++ MusicTheoryBridge
to access Python adaptive teacher capabilities.
"""

from typing import Dict, Any, Optional
import json

# Try to import adaptive teacher modules
try:
    from ml_framework.music_theory.adaptive_teacher import AdaptiveMusicTeacher
    from ml_framework.music_theory.exercise_generator import ExerciseGenerator
    from ml_framework.music_theory.explanation_engine import ExplanationEngine
    from ml_framework.music_theory.user_model import UserModel
    from ml_framework.music_theory.lesson_planner import LessonPlanner
    ADAPTIVE_TEACHER_AVAILABLE = True
except ImportError:
    ADAPTIVE_TEACHER_AVAILABLE = False
    AdaptiveMusicTeacher = None
    ExerciseGenerator = None
    ExplanationEngine = None
    UserModel = None
    LessonPlanner = None

# Global teacher instance (initialized on first use)
_teacher_instance: Optional[AdaptiveMusicTeacher] = None
_explanation_engine: Optional[ExplanationEngine] = None
_exercise_generator: Optional[ExerciseGenerator] = None
_user_model: Optional[UserModel] = None
_lesson_planner: Optional[LessonPlanner] = None


def _get_teacher():
    """Get or create adaptive teacher instance."""
    global _teacher_instance
    if _teacher_instance is None and ADAPTIVE_TEACHER_AVAILABLE:
        # Note: MusicTheoryBrain would be passed via pybind11 if available
        # For now, create teacher without brain (will use fallback)
        _teacher_instance = AdaptiveMusicTeacher(brain=None)
    return _teacher_instance


def _get_explanation_engine():
    """Get or create explanation engine instance."""
    global _explanation_engine
    if _explanation_engine is None and ADAPTIVE_TEACHER_AVAILABLE:
        _explanation_engine = ExplanationEngine()
    return _explanation_engine


def _get_exercise_generator():
    """Get or create exercise generator instance."""
    global _exercise_generator
    if _exercise_generator is None and ADAPTIVE_TEACHER_AVAILABLE:
        _exercise_generator = ExerciseGenerator()
    return _exercise_generator


def _get_user_model():
    """Get or create user model instance."""
    global _user_model
    if _user_model is None and ADAPTIVE_TEACHER_AVAILABLE:
        _user_model = UserModel()
    return _user_model


def _get_lesson_planner():
    """Get or create lesson planner instance."""
    global _lesson_planner
    if _lesson_planner is None and ADAPTIVE_TEACHER_AVAILABLE:
        _lesson_planner = LessonPlanner()
    return _lesson_planner


def explain_concept(concept: str, style: str, user_level: int) -> Dict[str, Any]:
    """
    Explain a concept with specified style.

    Args:
        concept: Concept name (e.g., "Perfect Fifth")
        style: Explanation style ("intuitive", "mathematical", "historical", "acoustic")
        user_level: User's current level (0-4: Beginner to Expert)

    Returns:
        Dictionary with "text" and "style" keys
    """
    engine = _get_explanation_engine()
    if engine:
        try:
            explanation = engine.explain(concept, style=style, user_level=user_level)
            return {
                "text": explanation.get("text", f"Explanation for {concept}"),
                "style": style
            }
        except Exception as e:
            return {
                "text": f"Error explaining concept: {str(e)}",
                "style": style
            }

    # Fallback
    return {
        "text": f"Explanation for {concept} ({style} style, level {user_level})",
        "style": style
    }


def generate_exercise(concept: str, user_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate exercise for a concept.

    Args:
        concept: Concept to practice
        user_profile: User profile dictionary with:
            - "userName": str
            - "currentLevel": int (0-4)
            - "masteredConcepts": List[str]
            - "strugglingConcepts": List[str]

    Returns:
        Dictionary with "question", "answer", "hints" keys
    """
    generator = _get_exercise_generator()
    if generator:
        try:
            exercise = generator.generate(concept, user_profile)
            return {
                "question": exercise.get("question", f"Practice {concept}"),
                "answer": exercise.get("answer", ""),
                "hints": exercise.get("hints", [])
            }
        except Exception as e:
            return {
                "question": f"Error generating exercise: {str(e)}",
                "answer": "",
                "hints": []
            }

    # Fallback
    return {
        "question": f"What is {concept}?",
        "answer": f"Answer for {concept}",
        "hints": [f"Think about {concept}"]
    }


def provide_feedback(exercise: Dict[str, Any], attempt: Dict[str, Any]) -> Dict[str, Any]:
    """
    Provide feedback on exercise attempt.

    Args:
        exercise: Exercise dictionary with "question", "answer", "hints"
        attempt: User attempt dictionary with:
            - "answer": str
            - "timeSpent": int (seconds)
            - "hintsUsed": int

    Returns:
        Dictionary with "is_correct", "explanation", "hint", "suggested_review" keys
    """
    # Simple feedback logic (can be enhanced with ML)
    user_answer = attempt.get("answer", "").strip().lower()
    correct_answer = exercise.get("answer", "").strip().lower()

    is_correct = user_answer == correct_answer or user_answer in correct_answer

    feedback = {
        "is_correct": is_correct,
        "explanation": "",
        "hint": "",
        "suggested_review": []
    }

    if is_correct:
        feedback["explanation"] = "Correct! Well done."
    else:
        feedback["explanation"] = f"Not quite. The correct answer is: {exercise.get('answer', 'N/A')}"
        if exercise.get("hints"):
            feedback["hint"] = exercise["hints"][0] if exercise["hints"] else ""
        feedback["suggested_review"] = [exercise.get("concept", "")]

    return feedback


def create_lesson_plan(concept: str, user_profile: Dict[str, Any]) -> str:
    """
    Create personalized lesson plan.

    Args:
        concept: Target concept
        user_profile: User profile dictionary

    Returns:
        JSON string with lesson plan
    """
    planner = _get_lesson_planner()
    if planner:
        try:
            lesson_plan = planner.create_plan(concept, user_profile)
            return json.dumps(lesson_plan)
        except Exception as e:
            return json.dumps({
                "error": f"Error creating lesson plan: {str(e)}",
                "concept": concept
            })

    # Fallback
    return json.dumps({
        "concept": concept,
        "lessons": [
            {
                "step": 1,
                "concept": concept,
                "rationale": "Introduction to concept",
                "estimatedMinutes": 10
            }
        ],
        "totalEstimatedHours": 0.2
    })
