"""
Test script to demonstrate what the Streamlit app does with all parameters.
Shows how 70+ parameters translate into a generated song.
"""

import json
from pathlib import Path


# Simulate what the Streamlit app would send
def create_test_song_params():
    """Create a comprehensive parameter set as the Streamlit app would."""

    return {
        # Basic Parameters
        "key": "C",
        "key_mode": "minor",
        "bpm": 90,
        "time_signature": "4/4",
        "genre": "cinematic",
        # Harmony & Chord Parameters
        "chord_style": "extended (9th, 11th, 13th)",
        "chord_complexity": 7,
        "chord_density": 6,
        "voice_leading": "smooth",
        "tension_level": 8,
        "dissonance": 4,
        "resolution_style": "delayed",
        "harmonic_rhythm": 5,
        # Rhythm & Groove Parameters
        "groove_style": "cinematic",
        "swing_amount": 0,
        "syncopation": 3,
        "rhythmic_complexity": 5,
        "subdivision": "1/8",
        "accent_pattern": "strong-weak",
        "ghost_notes": 2,
        "groove_pocket": 6,
        # Melody Parameters
        "melodic_range": "medium (2 octaves)",
        "melodic_complexity": 6,
        "contour": "arch",
        "interval_style": "mixed",
        "phrasing": "long legato",
        "ornamentation": 3,
        "repetition": 5,
        "development": 6,
        # Structure & Form Parameters
        "song_length": 32,
        "num_sections": 4,
        "intro_style": "fade in",
        "outro_style": "fade out",
        "section_transitions": "smooth",
        "repetition_variation": 5,
        "development_arc": "linear build",
        "bridge_probability": 30,
        # Dynamics & Expression Parameters
        "overall_volume": 75,
        "volume_envelope": "wave",
        "accent_strength": 5,
        "dynamic_range": 7,
        "vibrato": 3,
        "attack_time": "medium",
        "articulation": "legato",
        "expression_intensity": 6,
        # Instrumentation & Texture Parameters
        "num_voices": 6,
        "texture_density": 6,
        "layering_style": "polyphonic",
        "voice_roles": ["Melody", "Harmony", "Bass", "Rhythm", "Textural"],
        "instrument_balance": "balanced",
        "space_filling": 6,
        "frequency_range": "full spectrum",
        "stereo_width": 5,
        # Emotional & Psychological Parameters
        "valence": -5,  # Negative = sadness/grief
        "arousal": 6,  # Moderate-high = emotional intensity
        "intensity": 6,
        "energy_level": 5,
        "tension_release": 3,  # More tension than release
        "stability": 4,  # Somewhat unstable
        "predictability": 5,
        "emotional_complexity": 7,  # Complex emotions
        # Production & Sound Design Parameters
        "reverb_amount": 5,
        "reverb_type": "hall",
        "compression": 4,
        "eq_low": 2,
        "eq_mid": 0,
        "eq_high": -1,
        "saturation": 2,
        "filter_cutoff": 100,
        "delay_amount": 1,
        "chorus_flange": 0,
        # Style & Character Parameters
        "style_period": "contemporary",
        "cultural_influence": ["Western"],
        "performance_style": "cinematic",
        "humanization": 6,
        "imperfection": 4,
        "randomness": 2,
        "innovation": 5,
        "emotional_authenticity": 8,
    }


def demonstrate_parameter_effects():
    """Show how parameters translate to musical output."""

    print("=" * 80)
    print("KmiDi Music Generation - Parameter Translation Demo")
    print("=" * 80)
    print()

    # Test emotion: "I am feeling grief hidden as love with underlying tension"
    emotion_text = "I am feeling grief hidden as love with underlying tension"

    print("EMOTIONAL INTENT:")
    print(f'  "{emotion_text}"')
    print()

    params = create_test_song_params()

    print("=" * 80)
    print("PARAMETER ANALYSIS")
    print("=" * 80)
    print()

    # Harmony Analysis
    print("🎼 HARMONY & CHORD PROGRESSION:")
    print(f"  • Key: {params['key']} {params['key_mode']}")
    print(f"  • Chord Style: {params['chord_style']}")
    print(f"  • Complexity: {params['chord_complexity']}/10 (Complex, rich harmonies)")
    print(f"  • Tension Level: {params['tension_level']}/10 (High tension = unresolved, yearning)")
    print(f"  • Dissonance: {params['dissonance']}/10 (Some dissonance for emotional depth)")
    print(f"  • Resolution Style: {params['resolution_style']} (Delayed resolution = longing)")
    print()

    # Rhythm Analysis
    print("🥁 RHYTHM & GROOVE:")
    print(f"  • Tempo: {params['bpm']} BPM ({params['time_signature']})")
    print(f"  • Groove: {params['groove_style']} (Cinematic, atmospheric)")
    print(f"  • Syncopation: {params['syncopation']}/10 (Moderate, subtle off-beats)")
    print(f"  • Complexity: {params['rhythmic_complexity']}/10 (Balanced)")
    print(f"  • Subdivision: {params['subdivision']} notes (Smooth flow)")
    print()

    # Melody Analysis
    print("🎵 MELODY:")
    print(f"  • Range: {params['melodic_range']} (Expressive but controlled)")
    print(f"  • Complexity: {params['melodic_complexity']}/10 (Ornate, expressive)")
    print(f"  • Contour: {params['contour']} (Rising and falling = emotional arc)")
    print(f"  • Phrasing: {params['phrasing']} (Long, flowing lines)")
    print(f"  • Development: {params['development']}/10 (Evolves throughout)")
    print()

    # Structure Analysis
    print("📐 STRUCTURE & FORM:")
    print(f"  • Length: {params['song_length']} bars (4 sections)")
    print(f"  • Intro: {params['intro_style']} (Gradual entry)")
    print(f"  • Outro: {params['outro_style']} (Fading away)")
    print(f"  • Development Arc: {params['development_arc']} (Builds intensity)")
    print(f"  • Bridge Probability: {params['bridge_probability']}% (May include contrast)")
    print()

    # Emotional Analysis
    print("💫 EMOTIONAL TRANSLATION:")
    print(f"  • Valence: {params['valence']}/10 (Negative = sadness/grief)")
    print(f"  • Arousal: {params['arousal']}/10 (Moderate-high = emotional intensity)")
    print(f"  • Intensity: {params['intensity']}/10 (Strong emotional expression)")
    print(
        f"  • Tension-Release: {params['tension_release']}/10 (More tension = unresolved feeling)"
    )
    print(f"  • Stability: {params['stability']}/10 (Somewhat unstable = emotional uncertainty)")
    print(f"  • Complexity: {params['emotional_complexity']}/10 (Multiple layers of emotion)")
    print()

    # Instrumentation Analysis
    print("🎹 INSTRUMENTATION:")
    print(f"  • Voices: {params['num_voices']} instruments")
    print(f"  • Roles: {', '.join(params['voice_roles'])}")
    print(f"  • Texture: {params['texture_density']}/10 (Rich but not overwhelming)")
    print(f"  • Layering: {params['layering_style']} (Multiple independent voices)")
    print(f"  • Balance: {params['instrument_balance']} (Harmonious blend)")
    print()

    # Production Analysis
    print("🎛️ PRODUCTION & SOUND DESIGN:")
    print(
        f"  • Reverb: {params['reverb_amount']}/10 ({params['reverb_type']} - Spacious, cinematic)"
    )
    print(f"  • EQ: Low +{params['eq_low']}, Mid {params['eq_mid']}, High {params['eq_high']}")
    print(f"  • Dynamic Range: {params['dynamic_range']}/10 (Expressive dynamics)")
    print(f"  • Humanization: {params['humanization']}/10 (Natural, human feel)")
    print()

    print("=" * 80)
    print("EXPECTED MUSICAL OUTPUT")
    print("=" * 80)
    print()

    print("Based on these parameters, the generated song would have:")
    print()

    print("✓ Minor key with extended chords (9ths, 11ths) - rich, complex harmony")
    print("✓ High tension (8/10) with delayed resolution - creates 'longing' feeling")
    print("✓ Medium tempo (90 BPM) - contemplative, not rushed")
    print("✓ Cinematic style - atmospheric, emotionally evocative")
    print("✓ Negative valence + moderate-high arousal = grief with intensity")
    print("✓ Arch melodic contour - rises and falls like emotional journey")
    print("✓ Smooth voice leading - elegant, flowing movement")
    print("✓ Multiple voices in polyphonic texture - complex emotional layers")
    print("✓ Full spectrum frequency range - rich, complete soundscape")
    print("✓ Linear build development - gradual intensification")
    print()

    print("The result: A deeply emotional, cinematic piece that captures")
    print("'grief hidden as love with underlying tension' - a complex, layered")
    print("expression of sorrow with underlying warmth and unresolved yearning.")
    print()

    print("=" * 80)
    print(f"TOTAL PARAMETERS: {len(params)} detailed controls")
    print("=" * 80)

    # Save to file for reference
    output_file = Path("test_song_parameters.json")
    with open(output_file, "w") as f:
        json.dump(
            {
                "emotional_intent": emotion_text,
                "parameters": params,
                "summary": (
                    "Comprehensive parameter set demonstrating full control over song generation"
                ),
            },
            f,
            indent=2,
        )

    print(f"\n💾 Parameters saved to: {output_file}")
    print()


if __name__ == "__main__":
    demonstrate_parameter_effects()
