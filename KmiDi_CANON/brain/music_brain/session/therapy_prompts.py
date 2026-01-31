"""
Therapy Prompts Module - Enhanced feeling extraction using evidence-based therapy techniques.

Incorporates:
- Open-ended questions (Person-Centered, CBT)
- Miracle Question (Solution-Focused Therapy)
- Narrative Therapy questions
- Feeling identification methods
- Emotional granularity techniques
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import random


class TherapyApproach(Enum):
    """Different therapeutic approaches for question selection."""
    PERSON_CENTERED = "person_centered"  # Carl Rogers - open, non-directive
    SOLUTION_FOCUSED = "solution_focused"  # Miracle Question, future-oriented
    NARRATIVE = "narrative"  # Story reframing, externalization
    CBT = "cbt"  # Cognitive Behavioral - thought-feeling-behavior links
    GESTALT = "gestalt"  # Present-moment awareness, body sensations
    EMOTION_FOCUSED = "emotion_focused"  # Deep feeling exploration


@dataclass
class TherapyPrompt:
    """A single therapy prompt with metadata."""
    question: str
    approach: TherapyApproach
    purpose: str  # What feeling/insight this extracts
    follow_up_hints: List[str]  # Suggested follow-up questions
    feeling_keywords: List[str]  # Emotions this might surface


class TherapyPromptBank:
    """
    Bank of therapy prompts organized by approach and purpose.
    
    Based on evidence-based therapy techniques:
    - Person-Centered (Rogers): Open, non-judgmental exploration
    - Solution-Focused: Future-oriented, goal-focused
    - Narrative: Story reframing, externalization
    - CBT: Thought-feeling-behavior connections
    - Gestalt: Present-moment, body awareness
    - Emotion-Focused: Deep feeling exploration
    """
    
    # ============================================================================
    # OPEN-ENDED QUESTIONS (Person-Centered, General)
    # ============================================================================
    
    OPEN_ENDED_PROMPTS = [
        TherapyPrompt(
            question="What brings you here today?",
            approach=TherapyApproach.PERSON_CENTERED,
            purpose="Initial exploration, surface-level feeling identification",
            follow_up_hints=[
                "What's been weighing on you?",
                "What would you like to feel differently about?",
            ],
            feeling_keywords=["hurt", "pain", "struggle", "difficulty", "challenge"]
        ),
        TherapyPrompt(
            question="How does this situation make you feel?",
            approach=TherapyApproach.PERSON_CENTERED,
            purpose="Direct feeling identification",
            follow_up_hints=[
                "Where do you feel that in your body?",
                "When did you first notice this feeling?",
            ],
            feeling_keywords=["feel", "emotion", "sensation", "reaction"]
        ),
        TherapyPrompt(
            question="What emotion shows up most often for you right now?",
            approach=TherapyApproach.EMOTION_FOCUSED,
            purpose="Identify dominant emotional state",
            follow_up_hints=[
                "What triggers that emotion?",
                "What does that emotion need from you?",
            ],
            feeling_keywords=["often", "frequent", "dominant", "primary"]
        ),
        TherapyPrompt(
            question="Can you describe a time when you felt differently about this?",
            approach=TherapyApproach.CBT,
            purpose="Identify emotional patterns and triggers",
            follow_up_hints=[
                "What changed between then and now?",
                "What was different about that time?",
            ],
            feeling_keywords=["differently", "change", "before", "used to"]
        ),
        TherapyPrompt(
            question="What's the hardest part about this for you?",
            approach=TherapyApproach.PERSON_CENTERED,
            purpose="Identify core struggle and emotional pain point",
            follow_up_hints=[
                "What makes that part so difficult?",
                "What would make it easier?",
            ],
            feeling_keywords=["hardest", "difficult", "struggle", "pain"]
        ),
    ]
    
    # ============================================================================
    # MIRACLE QUESTION (Solution-Focused Therapy)
    # ============================================================================
    
    MIRACLE_QUESTION_PROMPTS = [
        TherapyPrompt(
            question="Imagine that tonight, while you sleep, a miracle occurs and your problem is solved. When you wake up tomorrow, what's the first thing you'll notice that's different?",
            approach=TherapyApproach.SOLUTION_FOCUSED,
            purpose="Envision desired future state, extract longing/desire",
            follow_up_hints=[
                "What would you feel like?",
                "What would others notice about you?",
                "What would you do differently?",
            ],
            feeling_keywords=["miracle", "different", "better", "relief", "peace"]
        ),
        TherapyPrompt(
            question="If you woke up tomorrow and everything felt right, what would that look like?",
            approach=TherapyApproach.SOLUTION_FOCUSED,
            purpose="Extract core longing and desired transformation",
            follow_up_hints=[
                "How would you know things were different?",
                "What would you feel in your body?",
            ],
            feeling_keywords=["right", "different", "better", "peace", "calm"]
        ),
        TherapyPrompt(
            question="What do you want to feel instead of what you're feeling now?",
            approach=TherapyApproach.SOLUTION_FOCUSED,
            purpose="Identify desired emotional state (core longing)",
            follow_up_hints=[
                "What would that feel like in your body?",
                "What would need to change for you to feel that?",
            ],
            feeling_keywords=["want", "instead", "desire", "longing", "wish"]
        ),
    ]
    
    # ============================================================================
    # NARRATIVE THERAPY QUESTIONS
    # ============================================================================
    
    NARRATIVE_PROMPTS = [
        TherapyPrompt(
            question="How would you title the story of what you're going through right now?",
            approach=TherapyApproach.NARRATIVE,
            purpose="Externalize the problem, extract emotional essence",
            follow_up_hints=[
                "What genre would this story be?",
                "Who are the main characters?",
            ],
            feeling_keywords=["story", "title", "narrative", "chapter"]
        ),
        TherapyPrompt(
            question="When did this problem first enter your story?",
            approach=TherapyApproach.NARRATIVE,
            purpose="Identify origin point and emotional timeline",
            follow_up_hints=[
                "What was happening in your life then?",
                "How has the story changed since then?",
            ],
            feeling_keywords=["first", "beginning", "started", "origin"]
        ),
        TherapyPrompt(
            question="Are there moments when this problem doesn't have as much power over you?",
            approach=TherapyApproach.NARRATIVE,
            purpose="Identify exceptions and moments of agency",
            follow_up_hints=[
                "What's different about those moments?",
                "What were you doing differently?",
            ],
            feeling_keywords=["power", "control", "different", "exception"]
        ),
        TherapyPrompt(
            question="If this feeling had a voice, what would it say?",
            approach=TherapyApproach.NARRATIVE,
            purpose="Externalize and personify the emotion",
            follow_up_hints=[
                "What does it want from you?",
                "What would it need to feel heard?",
            ],
            feeling_keywords=["voice", "say", "speak", "tell"]
        ),
    ]
    
    # ============================================================================
    # GESTALT / BODY AWARENESS QUESTIONS
    # ============================================================================
    
    BODY_AWARENESS_PROMPTS = [
        TherapyPrompt(
            question="Where do you feel this in your body?",
            approach=TherapyApproach.GESTALT,
            purpose="Somatic feeling identification",
            follow_up_hints=[
                "What does it feel like? (tight, heavy, empty, etc.)",
                "What color or texture would you give it?",
            ],
            feeling_keywords=["body", "feel", "sensation", "physical"]
        ),
        TherapyPrompt(
            question="If this feeling had a shape or color, what would it be?",
            approach=TherapyApproach.GESTALT,
            purpose="Metaphorical feeling expression",
            follow_up_hints=[
                "Where would it be located?",
                "How big or small is it?",
            ],
            feeling_keywords=["shape", "color", "texture", "image"]
        ),
        TherapyPrompt(
            question="What's happening in your body right now as you talk about this?",
            approach=TherapyApproach.GESTALT,
            purpose="Present-moment somatic awareness",
            follow_up_hints=[
                "What changes when you notice that?",
                "What does your body need right now?",
            ],
            feeling_keywords=["right now", "present", "body", "sensation"]
        ),
    ]
    
    # ============================================================================
    # EMOTION-FOCUSED / DEEP FEELING QUESTIONS
    # ============================================================================
    
    EMOTION_FOCUSED_PROMPTS = [
        TherapyPrompt(
            question="What's underneath this feeling?",
            approach=TherapyApproach.EMOTION_FOCUSED,
            purpose="Access deeper, primary emotions",
            follow_up_hints=[
                "What's the feeling beneath the feeling?",
                "What emotion is protecting you from feeling something else?",
            ],
            feeling_keywords=["underneath", "beneath", "below", "deeper"]
        ),
        TherapyPrompt(
            question="What does this emotion need from you?",
            approach=TherapyApproach.EMOTION_FOCUSED,
            purpose="Understand emotional needs and core longing",
            follow_up_hints=[
                "What would help this feeling feel heard?",
                "What action would honor this emotion?",
            ],
            feeling_keywords=["need", "want", "require", "deserve"]
        ),
        TherapyPrompt(
            question="If you could give this feeling a name, what would it be?",
            approach=TherapyApproach.EMOTION_FOCUSED,
            purpose="Precise emotional identification",
            follow_up_hints=[
                "What does that name tell you about what you're experiencing?",
                "How long has [name] been with you?",
            ],
            feeling_keywords=["name", "call", "identify", "label"]
        ),
        TherapyPrompt(
            question="What's the most vulnerable thing you could say about this?",
            approach=TherapyApproach.EMOTION_FOCUSED,
            purpose="Access core wound and deepest truth",
            follow_up_hints=[
                "What makes that so hard to say?",
                "What would happen if you said it?",
            ],
            feeling_keywords=["vulnerable", "hard", "difficult", "scary"]
        ),
    ]
    
    # ============================================================================
    # CBT / THOUGHT-FEELING-BEHAVIOR QUESTIONS
    # ============================================================================
    
    CBT_PROMPTS = [
        TherapyPrompt(
            question="What thoughts go through your mind when you feel this way?",
            approach=TherapyApproach.CBT,
            purpose="Link thoughts to feelings",
            follow_up_hints=[
                "How do those thoughts make you feel?",
                "What would you tell a friend who had those thoughts?",
            ],
            feeling_keywords=["thoughts", "mind", "think", "believe"]
        ),
        TherapyPrompt(
            question="What happens in your body when you have that thought?",
            approach=TherapyApproach.CBT,
            purpose="Connect cognitive and somatic experience",
            follow_up_hints=[
                "What emotion follows that physical sensation?",
                "What do you do when you feel that?",
            ],
            feeling_keywords=["body", "happens", "feel", "sensation"]
        ),
        TherapyPrompt(
            question="What do you do when you feel this way?",
            approach=TherapyApproach.CBT,
            purpose="Identify behavioral patterns and coping",
            follow_up_hints=[
                "Does that help or make it worse?",
                "What would you do differently if you could?",
            ],
            feeling_keywords=["do", "action", "behavior", "cope"]
        ),
    ]
    
    # ============================================================================
    # CORE WOUND / RESISTANCE QUESTIONS
    # ============================================================================
    
    CORE_WOUND_PROMPTS = [
        TherapyPrompt(
            question="What's the hardest thing to say about this?",
            approach=TherapyApproach.EMOTION_FOCUSED,
            purpose="Access core wound and resistance",
            follow_up_hints=[
                "What makes it so hard?",
                "What are you afraid would happen if you said it?",
            ],
            feeling_keywords=["hardest", "difficult", "afraid", "scary"]
        ),
        TherapyPrompt(
            question="What are you most afraid of feeling?",
            approach=TherapyApproach.EMOTION_FOCUSED,
            purpose="Identify emotional resistance and avoidance",
            follow_up_hints=[
                "What would happen if you let yourself feel that?",
                "What are you protecting yourself from?",
            ],
            feeling_keywords=["afraid", "fear", "avoid", "protect"]
        ),
        TherapyPrompt(
            question="What's at stake if you don't change this?",
            approach=TherapyApproach.SOLUTION_FOCUSED,
            purpose="Identify stakes and motivation for change",
            follow_up_hints=[
                "What would you lose?",
                "What would you gain?",
            ],
            feeling_keywords=["stake", "risk", "lose", "gain", "cost"]
        ),
        TherapyPrompt(
            question="What's holding you back from feeling what you want to feel?",
            approach=TherapyApproach.PERSON_CENTERED,
            purpose="Identify resistance and barriers (core resistance)",
            follow_up_hints=[
                "What would need to change?",
                "What are you afraid of?",
            ],
            feeling_keywords=["holding back", "prevent", "stop", "block"]
        ),
    ]
    
    # ============================================================================
    # TRANSFORMATION / DESIRED STATE QUESTIONS
    # ============================================================================
    
    TRANSFORMATION_PROMPTS = [
        TherapyPrompt(
            question="How do you want to feel when this is resolved?",
            approach=TherapyApproach.SOLUTION_FOCUSED,
            purpose="Identify desired transformation (core transformation)",
            follow_up_hints=[
                "What would that feel like in your body?",
                "How would you know you'd reached that state?",
            ],
            feeling_keywords=["want", "feel", "resolved", "different"]
        ),
        TherapyPrompt(
            question="What would need to happen for you to feel at peace with this?",
            approach=TherapyApproach.SOLUTION_FOCUSED,
            purpose="Identify conditions for emotional resolution",
            follow_up_hints=[
                "What's the smallest step toward that?",
                "What's already in place that supports that?",
            ],
            feeling_keywords=["peace", "resolved", "okay", "accept"]
        ),
        TherapyPrompt(
            question="What would it mean to you if you could feel differently about this?",
            approach=TherapyApproach.PERSON_CENTERED,
            purpose="Extract meaning and significance of change",
            follow_up_hints=[
                "What would become possible?",
                "How would your life be different?",
            ],
            feeling_keywords=["mean", "significance", "different", "possible"]
        ),
    ]
    
    # ============================================================================
    # METHODS
    # ============================================================================
    
    @classmethod
    def get_prompts_by_approach(cls, approach: TherapyApproach) -> List[TherapyPrompt]:
        """Get all prompts for a specific therapeutic approach."""
        all_prompts = (
            cls.OPEN_ENDED_PROMPTS +
            cls.MIRACLE_QUESTION_PROMPTS +
            cls.NARRATIVE_PROMPTS +
            cls.BODY_AWARENESS_PROMPTS +
            cls.EMOTION_FOCUSED_PROMPTS +
            cls.CBT_PROMPTS +
            cls.CORE_WOUND_PROMPTS +
            cls.TRANSFORMATION_PROMPTS
        )
        return [p for p in all_prompts if p.approach == approach]
    
    @classmethod
    def get_prompts_by_purpose(cls, purpose_keyword: str) -> List[TherapyPrompt]:
        """Get prompts that match a purpose keyword."""
        all_prompts = (
            cls.OPEN_ENDED_PROMPTS +
            cls.MIRACLE_QUESTION_PROMPTS +
            cls.NARRATIVE_PROMPTS +
            cls.BODY_AWARENESS_PROMPTS +
            cls.EMOTION_FOCUSED_PROMPTS +
            cls.CBT_PROMPTS +
            cls.CORE_WOUND_PROMPTS +
            cls.TRANSFORMATION_PROMPTS
        )
        purpose_lower = purpose_keyword.lower()
        return [p for p in all_prompts if purpose_lower in p.purpose.lower()]
    
    @classmethod
    def get_initial_prompt(cls) -> TherapyPrompt:
        """Get a good starting prompt for therapy session."""
        return random.choice([
            cls.OPEN_ENDED_PROMPTS[0],  # "What brings you here today?"
            cls.MIRACLE_QUESTION_PROMPTS[0],  # Miracle Question
            cls.EMOTION_FOCUSED_PROMPTS[0],  # "What emotion shows up most often?"
        ])
    
    @classmethod
    def get_core_wound_prompt(cls) -> TherapyPrompt:
        """Get a prompt specifically for accessing core wound."""
        return random.choice(cls.CORE_WOUND_PROMPTS)
    
    @classmethod
    def get_transformation_prompt(cls) -> TherapyPrompt:
        """Get a prompt for identifying desired transformation."""
        return random.choice(cls.TRANSFORMATION_PROMPTS)
    
    @classmethod
    def get_follow_up_prompt(cls, current_prompt: TherapyPrompt) -> Optional[TherapyPrompt]:
        """Get a logical follow-up prompt based on the current one."""
        # Simple logic: if current is about problem, follow with solution-focused
        if "problem" in current_prompt.question.lower() or "hurt" in current_prompt.question.lower():
            return random.choice(cls.MIRACLE_QUESTION_PROMPTS + cls.TRANSFORMATION_PROMPTS)
        
        # If current is about feeling, follow with body awareness or deeper exploration
        if "feel" in current_prompt.question.lower():
            return random.choice(cls.BODY_AWARENESS_PROMPTS + cls.EMOTION_FOCUSED_PROMPTS)
        
        # Default: return a related prompt from same approach
        same_approach = cls.get_prompts_by_approach(current_prompt.approach)
        if len(same_approach) > 1:
            return random.choice([p for p in same_approach if p != current_prompt])
        
        return None


# ============================================================================
# FEELING EXTRACTION METHODS
# ============================================================================

def extract_feeling_keywords(text: str, prompt: TherapyPrompt) -> List[str]:
    """
    Extract feeling-related keywords from text based on prompt context.
    
    Uses the prompt's feeling_keywords list to identify relevant emotional content.
    """
    text_lower = text.lower()
    found_keywords = []
    
    for keyword in prompt.feeling_keywords:
        if keyword in text_lower:
            found_keywords.append(keyword)
    
    return found_keywords


def identify_emotional_granularity(text: str) -> Dict[str, float]:
    """
    Identify emotional granularity - how specific vs. general the emotional language is.
    
    Returns:
        - specificity_score: 0.0 (vague) to 1.0 (very specific)
        - body_references: presence of somatic language
        - metaphor_use: presence of metaphorical language
    """
    # Specific emotion words (vs. vague "bad", "good")
    specific_emotions = [
        "grief", "rage", "nostalgia", "awe", "tenderness", "defiance",
        "dissociation", "confusion", "fear", "anxiety", "loneliness",
        "shame", "guilt", "envy", "jealousy", "resentment", "relief",
        "gratitude", "contentment", "euphoria", "melancholy"
    ]
    
    # Body/somatic references
    body_words = [
        "chest", "stomach", "throat", "shoulders", "head", "heart",
        "tight", "heavy", "empty", "numb", "tingling", "ache"
    ]
    
    # Metaphorical language
    metaphor_words = [
        "like", "as if", "as though", "feels like", "reminds me of",
        "color", "shape", "texture", "weight", "darkness", "light"
    ]
    
    text_lower = text.lower()
    
    specificity = sum(1 for emotion in specific_emotions if emotion in text_lower) / max(1, len(text.split()))
    body_refs = any(word in text_lower for word in body_words)
    metaphor_use = any(word in text_lower for word in metaphor_words)
    
    return {
        "specificity_score": min(1.0, specificity * 5),  # Scale up
        "body_references": body_refs,
        "metaphor_use": metaphor_use,
        "emotional_granularity": "high" if specificity > 0.1 or body_refs or metaphor_use else "low"
    }


def suggest_next_prompt(
    current_response: str,
    current_prompt: TherapyPrompt,
    session_depth: int = 0
) -> Optional[TherapyPrompt]:
    """
    Intelligently suggest the next prompt based on current response.
    
    Args:
        current_response: User's response to current prompt
        current_prompt: The prompt that was just answered
        session_depth: How many prompts have been asked (0 = first)
    
    Returns:
        Next suggested prompt, or None if session should conclude
    """
    # Early in session: explore the problem
    if session_depth == 0:
        return TherapyPromptBank.get_core_wound_prompt()
    
    # Mid-session: explore feelings deeper
    if session_depth == 1:
        granularity = identify_emotional_granularity(current_response)
        if granularity["emotional_granularity"] == "low":
            # Need more specific feeling language
            return random.choice(TherapyPromptBank.BODY_AWARENESS_PROMPTS + 
                                TherapyPromptBank.EMOTION_FOCUSED_PROMPTS)
        else:
            # Good granularity, move to transformation
            return TherapyPromptBank.get_transformation_prompt()
    
    # Later in session: focus on transformation and solution
    if session_depth >= 2:
        return random.choice(TherapyPromptBank.MIRACLE_QUESTION_PROMPTS + 
                           TherapyPromptBank.TRANSFORMATION_PROMPTS)
    
    # Default: follow-up from same approach
    return TherapyPromptBank.get_follow_up_prompt(current_prompt)


# ============================================================================
# CASUAL APPLICATION TO THERAPY PROMPTS
# ============================================================================

def create_casual_therapy_prompt(
    context: str = "",
    approach: Optional[TherapyApproach] = None
) -> str:
    """
    Create a casual, conversational therapy prompt.
    
    Makes therapy questions feel more natural and less clinical.
    """
    if approach is None:
        approach = random.choice(list(TherapyApproach))
    
    prompts = TherapyPromptBank.get_prompts_by_approach(approach)
    if not prompts:
        prompts = TherapyPromptBank.OPEN_ENDED_PROMPTS
    
    base_prompt = random.choice(prompts)
    
    # Make it more casual
    question = base_prompt.question
    
    # Add casual lead-ins
    casual_lead_ins = [
        "I'm curious...",
        "I wonder...",
        "Tell me...",
        "What's it like when...",
        "Help me understand...",
    ]
    
    # Only add lead-in if question doesn't already start conversationally
    if not question.lower().startswith(("tell me", "help me", "i'm", "i wonder")):
        if random.random() > 0.5:  # 50% chance
            question = f"{random.choice(casual_lead_ins)} {question.lower()}"
    
    return question


def extract_core_elements_from_response(response: str, prompt: TherapyPrompt) -> Dict[str, any]:
    """
    Extract core elements (wound, resistance, longing, stakes, transformation)
    from a therapy response.
    
    This is the bridge between therapy prompts and the DAiW intent schema.
    """
    elements = {
        "core_event": None,
        "core_resistance": None,
        "core_longing": None,
        "core_stakes": None,
        "core_transformation": None,
    }
    
    response_lower = response.lower()
    
    # Extract based on prompt purpose
    if "wound" in prompt.purpose or "hurt" in prompt.purpose:
        # This is likely about the core event
        elements["core_event"] = response
    
    if "resistance" in prompt.purpose or "holding back" in prompt.purpose:
        elements["core_resistance"] = response
    
    if "longing" in prompt.purpose or "want" in prompt.purpose or "desire" in prompt.purpose:
        elements["core_longing"] = response
    
    if "stake" in prompt.purpose or "risk" in prompt.purpose:
        elements["core_stakes"] = response
    
    if "transformation" in prompt.purpose or "resolved" in prompt.purpose:
        elements["core_transformation"] = response
    
    # Also extract feeling keywords
    feeling_keywords = extract_feeling_keywords(response, prompt)
    elements["feeling_keywords"] = feeling_keywords
    
    return elements

