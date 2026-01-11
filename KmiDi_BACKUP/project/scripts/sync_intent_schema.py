#!/usr/bin/env python3
"""
Schema Synchronization Utility

Validates that Python schema matches YAML schema and provides utilities
to export/import between formats.

Usage:
    python scripts/sync_intent_schema.py validate    # Check if schemas match
    python scripts/sync_intent_schema.py export      # Export Python schema to YAML
    python scripts/sync_intent_schema.py import       # Validate YAML against Python
"""

from __future__ import annotations

import json
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Set, Any

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from music_brain.session.intent_schema import (
    VALID_MOOD_PRIMARY_OPTIONS,
    VALID_IMAGERY_TEXTURE_OPTIONS,
    VALID_VULNERABILITY_SCALE_OPTIONS,
    VALID_NARRATIVE_ARC_OPTIONS,
    VALID_CORE_STAKES_OPTIONS,
    VALID_GENRE_OPTIONS,
    VALID_GROOVE_FEEL_OPTIONS,
    RULE_BREAKING_EFFECTS,
    HarmonyRuleBreak,
    RhythmRuleBreak,
    ArrangementRuleBreak,
    ProductionRuleBreak,
)


def load_yaml_schema() -> Dict[str, Any]:
    """Load the YAML schema file."""
    yaml_path = ROOT / "music_brain" / "data" / "song_intent_schema.yaml"
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def get_yaml_enums(yaml_data: Dict[str, Any]) -> Dict[str, List[str]]:
    """Extract enum lists from YAML schema."""
    enums = yaml_data.get("enums", {})
    return {
        "mood_primary_options": enums.get("mood_primary_options", []),
        "imagery_texture_options": enums.get("imagery_texture_options", []),
        "vulnerability_scale_options": enums.get("vulnerability_scale_options", []),
        "narrative_arc_options": enums.get("narrative_arc_options", []),
        "core_stakes_options": enums.get("core_stakes_options", []),
        "genre_options": enums.get("genre_options", []),
        "groove_feel_options": enums.get("groove_feel_options", []),
    }


def get_yaml_rule_breaking(yaml_data: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """Extract rule-breaking definitions from YAML schema."""
    enums = yaml_data.get("enums", {})
    rules = {}

    # Harmony rules
    harmony_rules = enums.get("harmony_rules_to_break", {})
    for key, value in harmony_rules.items():
        rules[key] = {
            "description": value.get("description", ""),
            "effect": value.get("effect", ""),
            "use_when": value.get("use_when", ""),
        }

    # Rhythm rules
    rhythm_rules = enums.get("rhythm_rules_to_break", {})
    for key, value in rhythm_rules.items():
        rules[key] = {
            "description": value.get("description", ""),
            "effect": value.get("effect", ""),
            "use_when": value.get("use_when", ""),
        }

    # Arrangement rules
    arrangement_rules = enums.get("arrangement_rules_to_break", {})
    for key, value in arrangement_rules.items():
        rules[key] = {
            "description": value.get("description", ""),
            "effect": value.get("effect", ""),
            "use_when": value.get("use_when", ""),
        }

    # Production rules
    production_rules = enums.get("production_rules_to_break", {})
    for key, value in production_rules.items():
        rules[key] = {
            "description": value.get("description", ""),
            "effect": value.get("effect", ""),
            "use_when": value.get("use_when", ""),
        }

    return rules


def compare_enums(python_enums: Dict[str, List[str]], yaml_enums: Dict[str, List[str]]) -> List[str]:
    """Compare Python and YAML enums, return list of discrepancies."""
    discrepancies = []

    enum_mapping = {
        "mood_primary_options": ("VALID_MOOD_PRIMARY_OPTIONS", python_enums.get("mood_primary_options", [])),
        "imagery_texture_options": ("VALID_IMAGERY_TEXTURE_OPTIONS", python_enums.get("imagery_texture_options", [])),
        "vulnerability_scale_options": ("VALID_VULNERABILITY_SCALE_OPTIONS", python_enums.get("vulnerability_scale_options", [])),
        "narrative_arc_options": ("VALID_NARRATIVE_ARC_OPTIONS", python_enums.get("narrative_arc_options", [])),
        "core_stakes_options": ("VALID_CORE_STAKES_OPTIONS", python_enums.get("core_stakes_options", [])),
        "genre_options": ("VALID_GENRE_OPTIONS", python_enums.get("genre_options", [])),
        "groove_feel_options": ("VALID_GROOVE_FEEL_OPTIONS", python_enums.get("groove_feel_options", [])),
    }

    for enum_name, (python_var, python_values) in enum_mapping.items():
        yaml_values = yaml_enums.get(enum_name, [])
        python_set = set(python_values)
        yaml_set = set(yaml_values)

        if python_set != yaml_set:
            missing_in_python = yaml_set - python_set
            missing_in_yaml = python_set - yaml_set

            if missing_in_python:
                discrepancies.append(
                    f"{enum_name}: Missing in Python ({python_var}): {sorted(missing_in_python)}"
                )
            if missing_in_yaml:
                discrepancies.append(
                    f"{enum_name}: Missing in YAML: {sorted(missing_in_yaml)}"
                )

    return discrepancies


def compare_rule_breaking(python_rules: Dict, yaml_rules: Dict) -> List[str]:
    """Compare rule-breaking definitions."""
    discrepancies = []

    python_keys = set(python_rules.keys())
    yaml_keys = set(yaml_rules.keys())

    missing_in_python = yaml_keys - python_keys
    missing_in_yaml = python_keys - yaml_keys

    if missing_in_python:
        discrepancies.append(f"Rules missing in Python: {sorted(missing_in_python)}")
    if missing_in_yaml:
        discrepancies.append(f"Rules missing in YAML: {sorted(missing_in_yaml)}")

    # Check fields for common rules
    common_keys = python_keys & yaml_keys
    for key in common_keys:
        python_rule = python_rules[key]
        yaml_rule = yaml_rules[key]

        required_fields = ["description", "effect", "use_when"]
        for field in required_fields:
            if field not in python_rule:
                discrepancies.append(f"{key}: Missing field '{field}' in Python")
            elif field not in yaml_rule:
                discrepancies.append(f"{key}: Missing field '{field}' in YAML")
            elif python_rule.get(field) != yaml_rule.get(field):
                discrepancies.append(f"{key}: Field '{field}' differs between Python and YAML")

    return discrepancies


def validate_schemas() -> bool:
    """Validate that Python and YAML schemas match."""
    print("Validating schema synchronization...")
    print("=" * 60)

    # Load YAML schema
    try:
        yaml_data = load_yaml_schema()
    except Exception as e:
        print(f"ERROR: Failed to load YAML schema: {e}")
        return False

    # Get Python enums
    python_enums = {
        "mood_primary_options": VALID_MOOD_PRIMARY_OPTIONS,
        "imagery_texture_options": VALID_IMAGERY_TEXTURE_OPTIONS,
        "vulnerability_scale_options": VALID_VULNERABILITY_SCALE_OPTIONS,
        "narrative_arc_options": VALID_NARRATIVE_ARC_OPTIONS,
        "core_stakes_options": VALID_CORE_STAKES_OPTIONS,
        "genre_options": VALID_GENRE_OPTIONS,
        "groove_feel_options": VALID_GROOVE_FEEL_OPTIONS,
    }

    # Get YAML enums
    yaml_enums = get_yaml_enums(yaml_data)

    # Compare enums
    enum_discrepancies = compare_enums(python_enums, yaml_enums)

    # Compare rule-breaking definitions
    yaml_rules = get_yaml_rule_breaking(yaml_data)
    rule_discrepancies = compare_rule_breaking(RULE_BREAKING_EFFECTS, yaml_rules)

    # Report results
    all_discrepancies = enum_discrepancies + rule_discrepancies

    if not all_discrepancies:
        print("✓ Schemas are synchronized!")
        print("\nEnum counts:")
        for enum_name, values in python_enums.items():
            print(f"  {enum_name}: {len(values)} options")
        print(f"\nRule-breaking rules: {len(RULE_BREAKING_EFFECTS)}")
        return True
    else:
        print("✗ Schema discrepancies found:\n")
        for disc in all_discrepancies:
            print(f"  - {disc}")
        return False


def export_python_to_yaml(output_path: Path | None = None):
    """Export Python schema constants to YAML format."""
    if output_path is None:
        output_path = ROOT / "music_brain" / "data" / "song_intent_schema_exported.yaml"

    export_data = {
        "schema_version": "1.0.0",
        "schema_name": "DAiW Song Intent Schema (Exported from Python)",
        "enums": {
            "mood_primary_options": VALID_MOOD_PRIMARY_OPTIONS,
            "imagery_texture_options": VALID_IMAGERY_TEXTURE_OPTIONS,
            "vulnerability_scale_options": VALID_VULNERABILITY_SCALE_OPTIONS,
            "narrative_arc_options": VALID_NARRATIVE_ARC_OPTIONS,
            "core_stakes_options": VALID_CORE_STAKES_OPTIONS,
            "genre_options": VALID_GENRE_OPTIONS,
            "groove_feel_options": VALID_GROOVE_FEEL_OPTIONS,
        },
    }

    with open(output_path, 'w') as f:
        yaml.dump(export_data, f, default_flow_style=False, sort_keys=False)

    print(f"Exported Python schema to: {output_path}")


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "validate":
        success = validate_schemas()
        sys.exit(0 if success else 1)
    elif command == "export":
        export_python_to_yaml()
    elif command == "import":
        # Import is same as validate - checks YAML against Python
        success = validate_schemas()
        sys.exit(0 if success else 1)
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
