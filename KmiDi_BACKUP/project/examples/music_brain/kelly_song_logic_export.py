#!/usr/bin/env python3
"""
Kelly Song - Complete Logic Pro Export Example

Demonstrates the full pipeline: Intent -> Emotion -> Music -> Mixer -> Logic Pro

This script shows how to use the Music Brain Emotion API to:
1. Define an emotional intent for a song
2. Map the emotion to musical parameters
3. Generate DAW mixer automation settings
4. Export everything to Logic Pro format

Song: "When I Found You Sleeping"
Artist: Kelly (fictional example)
Genre: Lo-fi bedroom emo / confessional acoustic
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from music_brain.emotion_api import MusicBrain, GeneratedMusic
from music_brain.session.intent_schema import (
    CompleteSongIntent,
    SongRoot,
    SongIntent,
    TechnicalConstraints,
)
from music_brain.daw.mixer_params import describe_mixer_params


def create_kelly_intent() -> CompleteSongIntent:
    """
    Define Kelly's complete song intent.

    This follows the three-phase interrogation model:
    - Phase 0: Core Wound/Desire
    - Phase 1: Emotional Intent
    - Phase 2: Technical Constraints
    """
    return CompleteSongIntent(
        title="When I Found You Sleeping",

        # Phase 0: The Core Wound
        song_root=SongRoot(
            core_event="Finding someone I loved after they chose to leave",
            core_resistance="Fear of making it about me, exploiting the loss",
            core_longing="To process grief without making it performative",
            core_stakes="Her memory deserves honesty, not poetry",
            core_transformation="Accept that grief doesn't resolve neatly"
        ),

        # Phase 1: Emotional Intent
        song_intent=SongIntent(
            mood_primary="Grief",
            mood_secondary_tension=0.3,  # Some nostalgia, some guilt
            vulnerability_scale="High",
            narrative_arc="Slow Reveal",  # Sounds like love until final line
            imagery_texture="Lo-fi bedroom, intimate, imperfect, soft morning light"
        ),

        # Phase 2: Technical Constraints
        technical_constraints=TechnicalConstraints(
            technical_key="F",
            technical_mode="major",
            technical_tempo_range=(78, 86),  # Slow, deliberate
            technical_genre="lo-fi bedroom emo",
            technical_groove_feel="Laid Back",
            technical_rule_to_break="HARMONY_ModalInterchange",
            rule_breaking_justification="Bbm (borrowed iv from F minor) makes hope feel earned and bittersweet; "
                                       "grief expressed through borrowed darkness that returns to major"
        )
    )


def main():
    print("=" * 70)
    print("KELLY SONG - 'When I Found You Sleeping'")
    print("Complete Logic Pro Export Pipeline")
    print("=" * 70)

    # Create the intent
    kelly_intent = create_kelly_intent()

    print(f"\n[Phase 0: Core Wound]")
    print(f"  Core Event: {kelly_intent.song_root.core_event}")
    print(f"  Core Stakes: {kelly_intent.song_root.core_stakes}")

    print(f"\n[Phase 1: Emotional Intent]")
    print(f"  Primary Mood: {kelly_intent.song_intent.mood_primary}")
    print(f"  Vulnerability: {kelly_intent.song_intent.vulnerability_scale}")
    print(f"  Narrative Arc: {kelly_intent.song_intent.narrative_arc}")
    print(f"  Imagery: {kelly_intent.song_intent.imagery_texture}")

    print(f"\n[Phase 2: Technical Constraints]")
    print(f"  Key: {kelly_intent.technical_constraints.technical_key} {kelly_intent.technical_constraints.technical_mode}")
    print(f"  Tempo: {kelly_intent.technical_constraints.technical_tempo_range[0]}-{kelly_intent.technical_constraints.technical_tempo_range[1]} BPM")
    print(f"  Genre: {kelly_intent.technical_constraints.technical_genre}")
    print(f"  Rule to Break: {kelly_intent.technical_constraints.technical_rule_to_break}")

    # Generate music from intent
    print(f"\n" + "-" * 70)
    print("GENERATING MUSIC FROM INTENT...")
    print("-" * 70)

    brain = MusicBrain()
    music = brain.generate_from_intent(kelly_intent)

    print(f"\n[Emotional Mapping]")
    print(f"  Primary Emotion: {music.emotional_state.primary_emotion}")
    print(f"  Valence: {music.emotional_state.valence}")
    print(f"  Arousal: {music.emotional_state.arousal}")

    print(f"\n[Musical Parameters]")
    print(f"  Tempo: {music.musical_params.tempo_suggested} BPM")
    print(f"  Tempo Range: {music.musical_params.tempo_min}-{music.musical_params.tempo_max}")
    print(f"  Timing Feel: {music.musical_params.timing_feel.value}")
    print(f"  Dissonance: {music.musical_params.dissonance:.0%}")
    print(f"  Space/Silence: {music.musical_params.space_probability:.0%}")

    print(f"\n[Mixer Settings]")
    print(f"  Description: {music.mixer_params.description}")
    print(f"  Tags: {', '.join(music.mixer_params.tags)}")
    print(f"\n  EQ:")
    print(f"    Presence: {music.mixer_params.eq_presence:+.1f}dB")
    print(f"    Air: {music.mixer_params.eq_air:+.1f}dB")
    print(f"    Low Mid: {music.mixer_params.eq_low_mid:+.1f}dB")
    print(f"\n  Dynamics:")
    print(f"    Compression: {music.mixer_params.compression_ratio:.1f}:1 @ {music.mixer_params.compression_threshold:.0f}dB")
    print(f"    Attack: {music.mixer_params.compression_attack:.0f}ms")
    print(f"    Release: {music.mixer_params.compression_release:.0f}ms")
    print(f"\n  Space:")
    print(f"    Reverb: {music.mixer_params.reverb_mix:.0%} mix, {music.mixer_params.reverb_decay:.1f}s decay")
    print(f"    Reverb Type: {music.mixer_params.reverb_type}")
    print(f"    Delay: {music.mixer_params.delay_mix:.0%} mix, {music.mixer_params.delay_time:.0f}ms")
    print(f"\n  Character:")
    print(f"    Saturation: {music.mixer_params.saturation:.0%} ({music.mixer_params.saturation_type})")
    print(f"    Filter Cutoff: {music.mixer_params.filter_cutoff:.0f}Hz")
    print(f"    Stereo Width: {music.mixer_params.stereo_width:.0%}")

    print(f"\n  Emotional Justification:")
    print(f"    {music.mixer_params.emotional_justification}")

    # Export to Logic Pro format
    print(f"\n" + "-" * 70)
    print("EXPORTING TO LOGIC PRO FORMAT...")
    print("-" * 70)

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    output_files = brain.export_to_logic(music, str(output_dir / "kelly_song"))

    print(f"\n[Exported Files]")
    for file_type, path in output_files.items():
        print(f"  {file_type}: {path}")

    # Also export full settings as text
    settings_path = output_dir / "kelly_song_mixer_settings.txt"
    with open(settings_path, 'w') as f:
        f.write(describe_mixer_params(music.mixer_params))
    print(f"  settings: {settings_path}")

    # Show next steps
    print(f"\n" + "=" * 70)
    print("NEXT STEPS FOR LOGIC PRO")
    print("=" * 70)

    print(f"""
1. IMPORT SETTINGS
   - Open Logic Pro and create a new project
   - Set tempo to {music.musical_params.tempo_suggested} BPM
   - Set key to {kelly_intent.technical_constraints.technical_key} {kelly_intent.technical_constraints.technical_mode}

2. APPLY EQ (Channel EQ)
   - Sub Bass (20-60Hz): {music.mixer_params.eq_sub_bass:+.1f}dB
   - Bass (60-250Hz): {music.mixer_params.eq_bass:+.1f}dB
   - Low Mid (250-500Hz): {music.mixer_params.eq_low_mid:+.1f}dB
   - Mid (500-2kHz): {music.mixer_params.eq_mid:+.1f}dB
   - High Mid (2-6kHz): {music.mixer_params.eq_high_mid:+.1f}dB
   - Presence (6-12kHz): {music.mixer_params.eq_presence:+.1f}dB
   - Air (12-20kHz): {music.mixer_params.eq_air:+.1f}dB

3. APPLY COMPRESSION
   - Ratio: {music.mixer_params.compression_ratio:.1f}:1
   - Threshold: {music.mixer_params.compression_threshold:.0f}dB
   - Attack: {music.mixer_params.compression_attack:.0f}ms
   - Release: {music.mixer_params.compression_release:.0f}ms
   - Knee: {music.mixer_params.compression_knee:.0f}dB

4. APPLY REVERB (Space Designer or ChromaVerb)
   - Type: {music.mixer_params.reverb_type.upper()}
   - Mix: {music.mixer_params.reverb_mix:.0%}
   - Decay: {music.mixer_params.reverb_decay:.1f}s
   - Predelay: {music.mixer_params.reverb_predelay:.0f}ms
   - Size: {music.mixer_params.reverb_size:.0%}

5. APPLY SATURATION (Tape plugins or Phat FX)
   - Type: {music.mixer_params.saturation_type.upper()}
   - Amount: {music.mixer_params.saturation:.0%}

6. CHORD PROGRESSION
   - F - C - Dm - Bbm (with modal interchange)
   - The Bbm is borrowed from F minor
   - This creates the bittersweet "grief through hope" quality

7. PRODUCTION AESTHETIC
   - Keep it lo-fi and imperfect
   - Room noise is acceptable (authenticity)
   - Pitch imperfections add vulnerability
   - The rolled-off highs create distance from reality

PHILOSOPHY:
"Interrogate Before Generate" - Every setting serves the emotional intent.
The audience doesn't hear "borrowed from parallel minor" -
they hear "that part made me cry."
""")

    print("=" * 70)

    # Also demonstrate the fluent API
    print("\n[BONUS: Fluent API Demo]")
    print("-" * 40)

    result = (brain.process("grief and loss, slow reveal of pain")
                   .map_to_emotion()
                   .map_to_music()
                   .with_tempo(82)
                   .map_to_mixer()
                   .get())

    print(f"  Emotion: {result['emotional_state']['primary_emotion']}")
    print(f"  Tempo: {result['musical_params']['tempo']} BPM")
    print(f"  Reverb: {result['mixer_params']['reverb']['mix']:.0f}%")
    print(f"  Description: {result['mixer_params']['metadata']['description']}")

    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
