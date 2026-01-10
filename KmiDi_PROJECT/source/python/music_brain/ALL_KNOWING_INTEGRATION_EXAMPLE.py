"""
Integration Example: All-Knowing Interactive Musical Customization System

This example demonstrates how all components work together.
"""

from music_brain.learning import UserPreferenceModel, PreferenceAnalyzer, PreferenceTracker
from music_brain.interactive import MultiParameterMorpher
from music_brain.intelligence import SuggestionEngine, ContextAnalyzer
from music_brain.adaptive import AdaptiveGenerator, FeedbackProcessor
from music_brain.editing import NaturalLanguageProcessor, FeedbackInterpreter, MidiEditor


def main():
    """Complete integration example."""

    print("=" * 70)
    print("All-Knowing Interactive Musical Customization System")
    print("=" * 70)

    # ====================================================================
    # Phase 1: User Preference Learning
    # ====================================================================
    print("\n[Phase 1] User Preference Learning")
    print("-" * 70)

    preference_model = UserPreferenceModel()
    tracker = PreferenceTracker(preference_model)

    # Simulate user interactions
    tracker.on_parameter_change("valence", 0.5, 0.7)
    tracker.on_parameter_change("intensity", 0.6, 0.8)
    tracker.on_emotion_selected("grief", valence=-0.7, arousal=0.4, intensity=0.6)

    # Get preferences
    prefs = preference_model.get_parameter_preferences()
    print(f"✓ Learned preferences: {list(prefs.keys())}")
    print(f"✓ Preferred valence range: {prefs.get('valence', {}).get('mean', 'N/A'):.2f}")

    # ====================================================================
    # Phase 2: Real-Time Parameter Adjustment
    # ====================================================================
    print("\n[Phase 2] Real-Time Parameter Adjustment")
    print("-" * 70)

    morpher = MultiParameterMorpher()
    morpher.set_parameter("valence", 0.5)
    morpher.set_parameter("arousal", 0.6)

    # Morph to new values
    morpher.set_target("valence", 0.8, duration=1.0, current_time=0.0)
    morpher.set_target("arousal", 0.4, duration=0.5, current_time=0.0)

    state = morpher.update(0.5)
    print(f"✓ Parameter morphing active: {morpher.has_active_morphs()}")
    print(f"✓ Current state: valence={state.get('valence'):.2f}, arousal={state.get('arousal'):.2f}")

    # ====================================================================
    # Phase 3: Intelligent Suggestions
    # ====================================================================
    print("\n[Phase 3] Intelligent Suggestions")
    print("-" * 70)

    suggestion_engine = SuggestionEngine(preference_model=preference_model)
    context_analyzer = ContextAnalyzer()

    current_state = {
        "emotion": "grief",
        "parameters": {
            "valence": -0.5,
            "arousal": 0.4,
            "intensity": 0.6,
            "tempo": 120,
        },
        "rule_breaks": []
    }

    context = context_analyzer.analyze(current_state)
    print(f"✓ Context analyzed: {context.emotion_category}, complexity={context.complexity_level}")

    suggestions = suggestion_engine.generate_suggestions(current_state, max_suggestions=3)
    print(f"✓ Generated {len(suggestions)} suggestions:")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"  {i}. {suggestion.title} (confidence: {suggestion.confidence:.2f})")

    # ====================================================================
    # Phase 4: Adaptive Generation
    # ====================================================================
    print("\n[Phase 4] Adaptive Generation")
    print("-" * 70)

    adaptive_generator = AdaptiveGenerator(preference_model=preference_model)
    feedback_processor = FeedbackProcessor(adaptive_generator)

    # Record feedback
    feedback_processor.process_explicit_feedback(
        parameters={"valence": -0.5, "arousal": 0.4},
        emotion="grief",
        thumbs_up=True
    )

    feedback_processor.process_implicit_feedback(
        original_parameters={"valence": -0.5, "arousal": 0.4},
        modified_parameters={"valence": -0.7, "arousal": 0.4},
        emotion="grief"
    )

    stats = adaptive_generator.get_statistics()
    print(f"✓ Adaptive generator statistics:")
    print(f"  - Total generations: {stats['total_generations']}")
    print(f"  - Acceptance rate: {stats['acceptance_rate']:.1%}")
    print(f"  - Parameter biases learned: {len(stats['parameter_biases'])}")

    # ====================================================================
    # Phase 6: Natural Language Feedback
    # ====================================================================
    print("\n[Phase 6] Natural Language Feedback Processing")
    print("-" * 70)

    nlp = NaturalLanguageProcessor()
    interpreter = FeedbackInterpreter(nlp)

    feedback_examples = [
        "I don't want so much chug",
        "bass line doesn't slap",
    ]

    for feedback in feedback_examples:
        result = interpreter.interpret_feedback(feedback, current_state)
        print(f"\n✓ Feedback: \"{feedback}\"")
        print(f"  Intent: {result.intent.value}")
        print(f"  Target: {result.target_element or 'general'}")
        print(f"  Changes: {result.parameter_changes}")
        print(f"  Confidence: {result.confidence:.2f}")

    # ====================================================================
    # Summary
    # ====================================================================
    print("\n" + "=" * 70)
    print("System Integration Complete")
    print("=" * 70)
    print("\nAll components are functional and ready for UI integration.")
    print("\nNext steps:")
    print("  - Integrate with C++ UI components (JUCE)")
    print("  - Connect PythonBridge for real-time communication")
    print("  - Add UI components for suggestions and natural language input")
    print("  - Implement MIDI editor UI (C++/JUCE)")


if __name__ == "__main__":
    main()
