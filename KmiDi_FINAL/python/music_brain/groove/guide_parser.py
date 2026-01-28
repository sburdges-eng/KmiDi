"""
Parser for 'Drum Programming Guide.md' to extract humanization rules.

This module bridges the gap between the human-readable Markdown guide and
the executable code in drum_humanizer.py. It parses specific sections
to build a structured ruleset.
"""

import re
from pathlib import Path
from typing import Dict, Any, List, Optional


class BaseGuideParser:
    """Base class for parsing markdown guides."""

    def __init__(self, guide_path: Path):
        self.guide_path = guide_path
        self.rules: Dict[str, Any] = {}

    def parse(self) -> Dict[str, Any]:
        """Main parsing method."""
        if not self.guide_path.exists():
            return self._get_fallback_rules()

        content = self.guide_path.read_text(encoding="utf-8")
        self._parse_content(content)
        return self.rules

    def _parse_content(self, content: str):
        """Override this to implement specific parsing logic."""
        pass

    def _get_fallback_rules(self) -> Dict[str, Any]:
        """Override this to provide default rules."""
        return {}


class DrumGuideParser(BaseGuideParser):
    """Parses the Drum Programming Guide markdown into structured rules."""

    def __init__(self, guide_path: Path):
        super().__init__(guide_path)
        self.rules = {
            "hihat": {},
            "snare": {},
            "kick": {},
            "genres": {}
        }

    def _parse_content(self, content: str):
        self._parse_hihat_rules(content)
        self._parse_snare_rules(content)
        self._parse_kick_rules(content)
        self._parse_genre_guidelines(content)

    def _parse_hihat_rules(self, content: str):
        """Extracts hi-hat velocity ranges and timing."""
        # Look for "Randomness" section
        # "Random variation ±5-10 velocity"
        vel_match = re.search(
            r"Random variation [±\+\-](\d+)-(\d+) velocity", content, re.IGNORECASE)
        if vel_match:
            self.rules["hihat"]["velocity_variation"] = (
                int(vel_match.group(1)), int(vel_match.group(2)))

        # "Random timing variation: ±10-20ms"
        time_match = re.search(
            r"Random timing variation: [±\+\-](\d+)-(\d+)ms", content, re.IGNORECASE)
        if time_match:
            self.rules["hihat"]["timing_variation_ms"] = (
                int(time_match.group(1)), int(time_match.group(2)))

    def _parse_snare_rules(self, content: str):
        """Extracts snare ghost note and main hit rules."""
        # Ghost notes: "Velocity: 25-45"
        ghost_match = re.search(
            r"Ghost Note Rules.*?Velocity: (\d+)-(\d+)", content, re.DOTALL | re.IGNORECASE)
        if ghost_match:
            self.rules["snare"]["ghost_velocity"] = (
                int(ghost_match.group(1)), int(ghost_match.group(2)))

        # Main hits: "Velocity range: 95-115"
        main_match = re.search(
            r"Main Snare Variation.*?Velocity range: (\d+)-(\d+)", content, re.DOTALL | re.IGNORECASE)
        if main_match:
            self.rules["snare"]["main_velocity"] = (
                int(main_match.group(1)), int(main_match.group(2)))

        # Timing: "Slight timing drift: ±5-10ms"
        time_match = re.search(
            r"Main Snare Variation.*?timing drift: [±\+\-](\d+)-(\d+)ms", content, re.DOTALL | re.IGNORECASE)
        if time_match:
            self.rules["snare"]["timing_variation_ms"] = (
                int(time_match.group(1)), int(time_match.group(2)))

    def _parse_kick_rules(self, content: str):
        """Extracts kick rules."""
        # Timing: "Timing: ±5ms"
        time_match = re.search(
            r"Kick Drum.*?Timing: [±\+\-](\d+)ms", content, re.DOTALL | re.IGNORECASE)
        if time_match:
            val = int(time_match.group(1))
            self.rules["kick"]["timing_variation_ms"] = (0, val)  # 0 to val

        # Velocity: "Velocity: Range of 85-110"
        vel_match = re.search(
            r"Kick Drum.*?Velocity: Range of (\d+)-(\d+)", content, re.DOTALL | re.IGNORECASE)
        if vel_match:
            self.rules["kick"]["velocity_range"] = (
                int(vel_match.group(1)), int(vel_match.group(2)))

    def _parse_genre_guidelines(self, content: str):
        """Extracts per-genre guidelines."""
        # Find the "Per-Genre Guidelines" section
        section_match = re.search(
            r"## Per-Genre Guidelines\n\n(.*?)$", content, re.DOTALL)
        if not section_match:
            return

        genre_section = section_match.group(1)
        # Split by "### GenreName"
        genres = re.split(r"###\s+", genre_section)

        for g in genres:
            if not g.strip():
                continue
            lines = g.strip().split('\n')
            name = lines[0].strip().lower()

            genre_rules = {
                "swing": 0.0,
                "timing_shift": 0.0,
                "notes": []
            }

            for line in lines[1:]:
                line = line.strip().lstrip('- ').strip()
                if not line:
                    continue
                genre_rules["notes"].append(line)

                # Parse specific keywords
                # "Heavy swing (55-62%)"
                swing_match = re.search(
                    r"swing.*?(\d+)-(\d+)%", line, re.IGNORECASE)
                if swing_match:
                    # Average the range and normalize to 0-1 (assuming 50% is 0.0, 75% is 1.0?
                    # Usually swing 50% = straight, 66% = triplet.
                    # Let's store raw percentage for now or normalize 50->0, 100->1)
                    # Standard DAW swing: 50% is straight.
                    avg_swing = (int(swing_match.group(1)) +
                                 int(swing_match.group(2))) / 2
                    # Normalize 50-100 to 0-1
                    genre_rules["swing"] = (avg_swing - 50) / 50.0

                # "snare slightly late (10-30ms)"
                late_match = re.search(
                    r"late.*?(\d+)-(\d+)ms", line, re.IGNORECASE)
                if late_match:
                    avg_late = (int(late_match.group(1)) +
                                int(late_match.group(2))) / 2
                    genre_rules["timing_shift"] = float(avg_late)

            self.rules["genres"][name] = genre_rules

    def _get_fallback_rules(self) -> Dict[str, Any]:
        """Hardcoded fallback if file is missing."""
        return {
            "hihat": {"velocity_variation": (5, 10), "timing_variation_ms": (10, 20)},
            "snare": {"ghost_velocity": (25, 45), "main_velocity": (95, 115), "timing_variation_ms": (5, 10)},
            "kick": {"velocity_range": (85, 110), "timing_variation_ms": (0, 5)},
            "genres": {
                "rock": {"swing": 0.0, "timing_shift": 0.0},
                "hip-hop": {"swing": 0.17, "timing_shift": 20.0}  # ~58% swing
            }
        }


class BassGuideParser(BaseGuideParser):
    """Parses the Bass Programming Guide markdown."""

    def __init__(self, guide_path: Path):
        super().__init__(guide_path)
        self.rules = {
            "timing": {},
            "velocity": {},
            "humanize": {}
        }

    def _parse_content(self, content: str):
        self._parse_timing(content)
        self._parse_velocity(content)
        self._parse_humanize(content)

    def _parse_timing(self, content: str):
        # "Behind the beat | 10-30ms late"
        behind_match = re.search(
            r"Behind the beat.*?(\d+)-(\d+)ms late", content, re.IGNORECASE)
        if behind_match:
            self.rules["timing"]["behind"] = (
                int(behind_match.group(1)), int(behind_match.group(2)))

        # "Ahead of the beat | 5-15ms early"
        ahead_match = re.search(
            r"Ahead of the beat.*?(\d+)-(\d+)ms early", content, re.IGNORECASE)
        if ahead_match:
            self.rules["timing"]["ahead"] = (
                int(ahead_match.group(1)), int(ahead_match.group(2)))

    def _parse_velocity(self, content: str):
        # "Root notes: Stronger (velocity 95-110)"
        root_match = re.search(
            r"Root notes.*?velocity (\d+)-(\d+)", content, re.IGNORECASE)
        if root_match:
            self.rules["velocity"]["root"] = (
                int(root_match.group(1)), int(root_match.group(2)))

        # "Ghost notes: Very soft (velocity 40-60)"
        ghost_match = re.search(
            r"Ghost notes.*?velocity (\d+)-(\d+)", content, re.IGNORECASE)
        if ghost_match:
            self.rules["velocity"]["ghost"] = (
                int(ghost_match.group(1)), int(ghost_match.group(2)))

    def _parse_humanize(self, content: str):
        # "Velocity: ±8 to ±12"
        vel_match = re.search(
            r"Velocity: [±\+\-](\d+) to [±\+\-](\d+)", content, re.IGNORECASE)
        if vel_match:
            self.rules["humanize"]["velocity_range"] = (
                int(vel_match.group(1)), int(vel_match.group(2)))

    def _get_fallback_rules(self) -> Dict[str, Any]:
        return {
            "timing": {"behind": (10, 30), "ahead": (5, 15)},
            "velocity": {"root": (95, 110), "ghost": (40, 60)},
            "humanize": {"velocity_range": (8, 12)}
        }


class GuitarGuideParser(BaseGuideParser):
    """Parses the Guitar Programming Guide markdown."""

    def __init__(self, guide_path: Path):
        super().__init__(guide_path)
        self.rules = {
            "strumming": {},
            "velocity": {}
        }

    def _parse_content(self, content: str):
        self._parse_strumming(content)
        self._parse_velocity(content)

    def _parse_strumming(self, content: str):
        # "Down strum | Low to high | 20-50ms total"
        strum_match = re.search(
            r"Down strum.*?(\d+)-(\d+)ms total", content, re.IGNORECASE)
        if strum_match:
            self.rules["strumming"]["total_duration"] = (
                int(strum_match.group(1)), int(strum_match.group(2)))

        # "Stagger notes by 5-10ms each"
        stagger_match = re.search(
            r"Stagger notes by (\d+)-(\d+)ms", content, re.IGNORECASE)
        if stagger_match:
            self.rules["strumming"]["note_stagger"] = (
                int(stagger_match.group(1)), int(stagger_match.group(2)))

    def _parse_velocity(self, content: str):
        # "Accents: 100 vs 70" - heuristic search for accent patterns
        # "Beat: 1 ... Vel: 100"
        pass  # Complex to parse table, maybe just look for ranges if available

    def _get_fallback_rules(self) -> Dict[str, Any]:
        return {
            "strumming": {"total_duration": (20, 50), "note_stagger": (5, 10)},
            "velocity": {}
        }

class EQGuideParser(BaseGuideParser):
    """Parses the EQ Deep Dive Guide."""
    
    def __init__(self, guide_path: Path):
        super().__init__(guide_path)
        self.rules = {
            "instruments": {}
        }

    def _parse_content(self, content: str):
        # Parse "What Lives Where" table
        # | Kick drum | Sub: 50-60Hz, Body: 80-100Hz, Click: 3-5kHz |
        
        # Find table section
        table_match = re.search(r"## What Lives Where\n\n(.*?)\n\n", content, re.DOTALL)
        if table_match:
            table_text = table_match.group(1)
            rows = table_text.strip().split('\n')
            for row in rows:
                if "|" not in row or "Instrument" in row or "---" in row:
                    continue
                
                parts = [p.strip() for p in row.split('|') if p.strip()]
                if len(parts) >= 2:
                    inst = parts[0].lower()
                    desc = parts[1]
                    
                    # Parse frequency ranges: "Sub: 50-60Hz"
                    ranges = {}
                    # Regex for "Label: RangeHz"
                    # e.g. "Sub: 50-60Hz" or "Click: 3-5kHz"
                    matches = re.finditer(r"([A-Za-z]+):\s*([\d\.\-]+)(Hz|kHz)", desc)
                    for m in matches:
                        label = m.group(1).lower()
                        val_str = m.group(2)
                        unit = m.group(3)
                        
                        # Convert to Hz
                        try:
                            if '-' in val_str:
                                low, high = val_str.split('-')
                                low = float(low)
                                high = float(high)
                            else:
                                low = high = float(val_str)
                                
                            if unit == 'kHz':
                                low *= 1000
                                high *= 1000
                                
                            ranges[label] = (low, high)
                        except ValueError:
                            continue
                            
                    self.rules["instruments"][inst] = ranges

    def _get_fallback_rules(self) -> Dict[str, Any]:
        return {
            "instruments": {
                "kick drum": {"sub": (50, 60), "click": (3000, 5000)},
                "snare": {"body": (150, 200), "crack": (1000, 2000)}
            }
        }


class CompressionGuideParser(BaseGuideParser):
    """Parses the Compression Deep Dive Guide."""
    
    def __init__(self, guide_path: Path):
        super().__init__(guide_path)
        self.rules = {
            "ratios": {},
            "attack": {}
        }

    def _parse_content(self, content: str):
        self._parse_ratios(content)
        self._parse_attack(content)

    def _parse_ratios(self, content: str):
        # | 6:1 | Aggressive | Drums, punchy sounds |
        # Regex to find ratio lines
        matches = re.finditer(r"\|\s*(\d+):1\s*\|\s*([^|]+)\s*\|\s*([^|]+)\|", content)
        for m in matches:
            ratio = int(m.group(1))
            desc = m.group(2).strip()
            uses = m.group(3).strip().lower()
            
            # Map uses to ratio
            if "drums" in uses:
                self.rules["ratios"]["drums"] = ratio
            if "vocals" in uses:
                self.rules["ratios"]["vocals"] = ratio
            if "guitars" in uses:
                self.rules["ratios"]["guitars"] = ratio

    def _parse_attack(self, content: str):
        # | Fast (0-10ms) | ...
        matches = re.finditer(r"\|\s*([A-Za-z]+)\s*\((\d+)-(\d+)ms\)\s*\|", content)
        for m in matches:
            speed = m.group(1).lower()
            low = int(m.group(2))
            high = int(m.group(3))
            self.rules["attack"][speed] = (low, high)

    def _get_fallback_rules(self) -> Dict[str, Any]:
        return {
            "ratios": {"drums": 6, "vocals": 3, "guitars": 3},
            "attack": {"fast": (0, 10), "slow": (30, 100)}
        }
