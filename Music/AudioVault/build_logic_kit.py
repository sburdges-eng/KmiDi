#!/usr/bin/env python3
"""
Build Logic Pro Kit
Creates EXS24 sampler instrument from refined samples
"""

import json
from pathlib import Path

VAULT_DIR = Path.home() / "Music" / "AudioVault"
REFINED_DIR = VAULT_DIR / "refined"
KITS_DIR = VAULT_DIR / "kits"
LOGIC_SAMPLER_DIR = Path.home() / "Music" / "Audio Music Apps" / "Sampler Instruments"


# GM Drum Mapping (compatible with MPK mini 3)
GM_DRUM_MAP = {
    36: "Kick",
    38: "Snare",
    40: "Snare",  # Rim shot
    42: "HiHat_Closed",
    44: "HiHat_Closed",  # Pedal
    46: "HiHat_Open",
    49: "Crash",
    51: "Crash",  # Ride
    41: "Tom_Low",
    43: "Tom_Low",
    45: "Tom_Mid",
    47: "Tom_Mid",
    48: "Tom_High",
}


def find_sample_for_voice(voice_name):
    """Find best sample for a drum voice"""
    # Search in refined directory
    pattern = f"*{voice_name}*.wav"
    matches = list(REFINED_DIR.rglob(pattern))

    if matches:
        # Return first match (could be improved with ranking)
        return matches[0]
    return None


def create_exs24_mapping(kit_name="LoFi_Bedroom_Kit"):
    """Create EXS24 instrument mapping"""
    mapping = {
        "kit_name": kit_name,
        "format": "EXS24",
        "zones": []
    }

    print(f"\n🔍 Building {kit_name}...\n")

    for midi_note, voice_name in GM_DRUM_MAP.items():
        sample_path = find_sample_for_voice(voice_name)

        if sample_path:
            zone = {
                "midi_note": midi_note,
                "note_name": voice_name,
                "sample_path": str(sample_path),
                "relative_path": str(sample_path.relative_to(REFINED_DIR)),
                "velocity_min": 1,
                "velocity_max": 127,
                "pitch": 0,  # No pitch shift
                "volume": 0,  # 0dB
            }
            mapping["zones"].append(zone)
            print(f"✅ {midi_note} ({voice_name}): {sample_path.name}")
        else:
            print(f"⚠️  {midi_note} ({voice_name}): No sample found")

    return mapping


def save_mapping(mapping, kit_name):
    """Save mapping to JSON and text guide"""
    # Create kits directory
    KITS_DIR.mkdir(parents=True, exist_ok=True)
    LOGIC_SAMPLER_DIR.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_file = KITS_DIR / f"{kit_name}.json"
    with open(json_file, 'w') as f:
        json.dump(mapping, f, indent=2)

    # Save text guide for Logic Pro
    guide_file = LOGIC_SAMPLER_DIR / f"{kit_name}_GUIDE.txt"
    with open(guide_file, 'w') as f:
        f.write(f"# {kit_name} - Logic Pro Setup Guide\n\n")
        f.write("## Sample Mapping\n\n")

        for zone in mapping["zones"]:
            f.write(f"MIDI {zone['midi_note']}: {zone['note_name']}\n")
            f.write(f"   {zone['relative_path']}\n\n")

        f.write("\n## Logic Pro Instructions\n\n")
        f.write("1. Open Logic Pro\n")
        f.write("2. Create Software Instrument track\n")
        f.write("3. Load 'Sampler' or 'Quick Sampler' plugin\n")
        f.write("4. Drag samples from AudioVault/refined/ onto MIDI notes\n")
        f.write("5. Use the mapping above\n")
        f.write(f"6. Save as: {kit_name}\n\n")

        f.write("## MPK mini 3 Pads Mapping\n\n")
        f.write("Pad 1 (C1/36):  Kick\n")
        f.write("Pad 2 (C#1/37): <unmapped>\n")
        f.write("Pad 3 (D1/38):  Snare\n")
        f.write("Pad 4 (D#1/39): <unmapped>\n")
        f.write("Pad 5 (E1/40):  Snare (Rim)\n")
        f.write("Pad 6 (F1/41):  Tom Low\n")
        f.write("Pad 7 (F#1/42): HiHat Closed\n")
        f.write("Pad 8 (G1/43):  Tom Low\n\n")

    return json_file, guide_file


def main():
    """Build Logic Pro kit"""
    print("=" * 60)
    print("LOGIC PRO KIT BUILDER")
    print("=" * 60)

    if not REFINED_DIR.exists():
        print(f"❌ Refined directory not found: {REFINED_DIR}")
        print("   Run audio_refinery.py first")
        return

    # Create mapping
    kit_name = "LoFi_Bedroom_Kit"
    mapping = create_exs24_mapping(kit_name)

    if not mapping["zones"]:
        print("\n❌ No samples found to build kit")
        return

    # Save
    json_file, guide_file = save_mapping(mapping, kit_name)

    print("\n" + "=" * 60)
    print(f"✅ KIT BUILT: {kit_name}")
    print("=" * 60)
    print(f"\nZones: {len(mapping['zones'])}")
    print(f"JSON: {json_file}")
    print(f"Guide: {guide_file}")
    print(f"\n📖 Open guide: open '{guide_file}'")
    print(f"\n🎹 MIDI range: C1 (36) - C2 (48)")
    print(f"   Compatible with MPK mini 3 pads")
    print()


if __name__ == "__main__":
    main()
