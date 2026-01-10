# Critical Broken Code - Line-by-Line Analysis

## 1. BROKEN: Chord to Degree Conversion
**File:** `/music_brain/structure/progression.py`
**Issue:** Returns 0 for chromatic chords, causing false matches

### Current Broken Code:
```python
def chord_to_degree(root_pc: int, key_root_pc: int):
    scale_pcs = _scale_degrees_for_key(key_root_pc)  # MAJOR ONLY!
    if root_pc in scale_pcs:
        return scale_pcs.index(root_pc) + 1
    return 0  # chromatic passing  <-- PROBLEM: 0 is ambiguous
```

### Why It's Broken:
1. The value `0` is used for "non-diatonic" but also means "no chord"
2. Later code does `if a != 0` treating 0 as "ignore"
3. A progression `[0, 0, 0, 0]` matches EVERY pattern

### Fix Required:
```python
def chord_to_degree(root_pc: int, key_root_pc: int, key_mode='major'):
    if key_mode == 'major':
        scale_steps = [0, 2, 4, 5, 7, 9, 11]
    elif key_mode == 'minor':
        scale_steps = [0, 2, 3, 5, 7, 8, 10]  # Natural minor
    else:
        # Add other modes
        raise ValueError(f"Unknown mode: {key_mode}")
    
    scale_pcs = [(key_root_pc + s) % 12 for s in scale_steps]
    
    if root_pc in scale_pcs:
        degree = scale_pcs.index(root_pc) + 1
        return degree
    
    # Return negative for chromatic (distinguishable from diatonic)
    # Check common chromatic chords
    chromatic_map = {
        (key_root_pc + 1) % 12: -2,   # bII (Neapolitan)
        (key_root_pc + 6) % 12: -4.5,  # bV (tritone sub)
        (key_root_pc + 10) % 12: -7,   # bVII (borrowed)
    }
    
    return chromatic_map.get(root_pc, -99)  # -99 = unknown chromatic
```

---

## 2. BROKEN: Timing Map Semantics
**File:** `/music_brain/groove/genre_templates.py` (older versions)
**Issue:** timing_map values have no clear meaning

### Current Broken Code (from old version):
```python
"timing_map": [0.1, 0.4, 0.2, 0.3, 0.5, 0.3, 0.4, 0.2,
               0.1, 0.3, 0.6, 0.3, 0.4, 0.2, 0.3, 0.2]
# What do these numbers mean??
```

### Problems:
1. Values range 0.1-0.6 but unit is undefined
2. Don't sum to 1.0 (not probabilities)
3. Not in ticks (would be integers)
4. Not in milliseconds (too small)

### Fix Required:
```python
# REPLACE timing_map with clear semantics:
"timing_density": [  # Probability of note on each 16th (0.0-1.0)
    0.95, 0.30, 0.40, 0.35,  # Beat 1
    0.50, 0.35, 0.45, 0.40,  # Beat 2
    0.90, 0.35, 0.50, 0.40,  # Beat 3 (snare)
    0.50, 0.40, 0.45, 0.35,  # Beat 4
],
"timing_offset": [  # Push/pull from grid in ticks (at 480 PPQ)
    0, -3, 5, -2,   # Negative = ahead, Positive = behind
    0, -2, 6, -3,
    0, -3, 5, -2,
    0, -2, 4, -3,
],
```

---

## 3. BROKEN: Progression Pattern Matching
**File:** `/music_brain/structure/progression.py`

### Current Broken Code:
```python
def match_progression_families(degrees, families, tolerance=1):
    matches = []
    for name, pattern in families.items():
        for i in range(len(degrees) - len(pattern) + 1):
            window = degrees[i:i + len(pattern)]
            mismatches = sum(1 for a, b in zip(window, pattern) 
                           if a != b and a != 0)  # <-- BROKEN LOGIC
            if mismatches <= tolerance:
                matches.append((name, i, mismatches))
    return matches
```

### Why It's Broken:
1. `a != 0` means chromatic chords NEVER count as mismatches
2. Tolerance of 1 on 4-chord pattern = 75% match is "success"
3. No confidence score, just binary match

### Fix Required:
```python
def match_progression_families(degrees, families, min_confidence=0.75):
    matches = []
    
    for name, pattern in families.items():
        pattern_len = len(pattern)
        
        for i in range(len(degrees) - pattern_len + 1):
            window = degrees[i:i + pattern_len]
            
            # Calculate match score
            exact_matches = 0
            chromatic_penalties = 0
            
            for observed, expected in zip(window, pattern):
                if observed == expected:
                    exact_matches += 1
                elif observed < 0:  # Chromatic chord
                    # Check if it's a common substitution
                    if is_valid_substitution(observed, expected):
                        exact_matches += 0.5  # Partial credit
                    else:
                        chromatic_penalties += 1
                # else: different diatonic chord
            
            # Calculate confidence
            score = exact_matches / pattern_len
            score -= (chromatic_penalties * 0.1)  # Penalty for non-functional chromatics
            
            if score >= min_confidence:
                matches.append({
                    'name': name,
                    'start_index': i,
                    'confidence': score,
                    'exact_matches': exact_matches,
                    'pattern': pattern,
                    'observed': window
                })
    
    # Sort by confidence
    return sorted(matches, key=lambda x: x['confidence'], reverse=True)
```

---

## 4. BROKEN: Duplicate Progression Patterns
**File:** `/music_brain/structure/progression.py`

### Current Broken Code:
```python
PROGRESSION_FAMILIES = {
    "pop_1546": [1, 5, 6, 4],
    "pop_1564": [1, 5, 6, 4],  # EXACT DUPLICATE!
}
```

### Fix Required:
```python
PROGRESSION_FAMILIES = {
    "pop_1546": [1, 5, 4, 6],  # Correct: I-V-IV-vi
    "pop_1564": [1, 5, 6, 4],  # Correct: I-V-vi-IV (Axis)
}
```

---

## 5. BROKEN: Key Parsing
**File:** `/music_brain/structure/chord.py`

### Current Broken Code:
```python
def parse_key(key_name: str):
    name = key_name.split()[0]  # "C major" -> "C"
    key_map = {"C": 0, "C#": 1, "Db": 1, ...}
    return key_map.get(name, 0)  # Default to C??
```

### Problems:
1. Doesn't handle "Cmaj", "CM", "C", "c", "Cm"
2. No minor/major detection
3. Silent failure (returns 0)

### Fix Required:
```python
import re

def parse_key(key_name: str):
    """
    Parse key from various formats:
    'C major', 'C', 'Cmaj', 'CM', 'C Major'
    'A minor', 'Am', 'Amin', 'a'
    """
    key_name = key_name.strip()
    
    # Regex to match key patterns
    pattern = r'^([A-Ga-g])([#b]?)(\s*)(maj|major|min|minor|m|M)?'
    match = re.match(pattern, key_name, re.IGNORECASE)
    
    if not match:
        raise ValueError(f"Cannot parse key: {key_name}")
    
    note, accidental, _, mode_str = match.groups()
    
    # Determine root
    note = note.upper()
    root_map = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    
    if note not in root_map:
        raise ValueError(f"Invalid note: {note}")
    
    root = root_map[note]
    
    # Apply accidental
    if accidental == '#':
        root = (root + 1) % 12
    elif accidental == 'b':
        root = (root - 1) % 12
    
    # Determine mode
    mode = 'major'  # Default
    if mode_str:
        mode_lower = mode_str.lower()
        if mode_lower in ['min', 'minor', 'm']:
            mode = 'minor'
        # Lowercase note often implies minor
    elif note.islower():
        mode = 'minor'
    
    return {
        'root': root,
        'mode': mode,
        'original': key_name
    }
```

---

## 6. BROKEN: Beat Density Division by Zero
**File:** `/music_brain/audio/feel.py`

### Current Broken Code:
```python
beat_density = len(beat_times) / (beat_times[-1] - beat_times[0]) if len(beat_times) > 1 else 0.0
```

### Problem:
If `beat_times[-1] == beat_times[0]` (two beats at same time), ZeroDivisionError

### Fix Required:
```python
if len(beat_times) > 1:
    time_span = beat_times[-1] - beat_times[0]
    if time_span > 0:
        beat_density = len(beat_times) / time_span
    else:
        # Beats at same time - probably an error
        beat_density = 0.0
        logger.warning(f"Beat times have zero span: {beat_times}")
else:
    beat_density = 0.0
```

---

## 7. BROKEN: Librosa Tempo Return Type
**File:** `/music_brain/audio/feel.py`

### Current Broken Code:
```python
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
"tempo_bpm": float(tempo),  # tempo might be array!
```

### Problem:
Recent librosa versions return tempo as array, not scalar

### Fix Required:
```python
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

# Handle both old (scalar) and new (array) librosa versions
if isinstance(tempo, np.ndarray):
    if tempo.size > 0:
        tempo_bpm = float(tempo[0])
    else:
        tempo_bpm = 120.0  # Default
else:
    tempo_bpm = float(tempo)

"tempo_bpm": tempo_bpm,
```

---

## 8. MISSING: Instrument Assignment
**File:** `/music_brain/groove/templates.py`

### Current Problem:
```python
"velocity_curve": [100, 65, 75, 60, ...]  # For WHICH instrument??
```

### Fix Required:
```python
"velocity_curves": {
    "kick": [100, 0, 0, 0, 85, 0, 0, 0, 105, 0, 0, 0, 85, 0, 0, 0],
    "snare": [0, 0, 0, 0, 0, 0, 0, 0, 110, 0, 0, 0, 0, 0, 0, 0],
    "hihat": [70, 65, 75, 60, 70, 65, 75, 60, 70, 65, 75, 60, 70, 65, 75, 60],
    "ride": None,  # Not used in this pattern
},
"default_velocity": 80,  # For unspecified instruments
```

---

## 9. MISSING: Template Validation
**File:** Referenced but not found

### Expected Function:
```python
def validate_template(template: Dict[str, Any]) -> List[str]:
    """
    Validate template structure and values.
    Returns list of issues (empty if valid).
    """
    issues = []
    
    # Check required fields
    required = ['ppq', 'grid', 'swing_ratio']
    for field in required:
        if field not in template:
            issues.append(f"Missing required field: {field}")
    
    # Validate swing_ratio
    if 'swing_ratio' in template:
        ratio = template['swing_ratio']
        if not (0.5 <= ratio <= 0.75):
            issues.append(f"Invalid swing_ratio: {ratio} (must be 0.5-0.75)")
    
    # Validate velocity values
    if 'velocity_curve' in template:
        for i, vel in enumerate(template['velocity_curve']):
            if not (0 <= vel <= 127):
                issues.append(f"Invalid velocity at position {i}: {vel}")
    
    # Validate timing_density (if using new format)
    if 'timing_density' in template:
        for i, density in enumerate(template['timing_density']):
            if not (0.0 <= density <= 1.0):
                issues.append(f"Invalid timing_density at position {i}: {density}")
    
    # Validate grid size
    if 'grid' in template:
        if template['grid'] not in [8, 16, 32]:
            issues.append(f"Unusual grid size: {template['grid']}")
    
    # Validate pocket offsets
    if 'pocket' in template:
        for inst, offset in template['pocket'].items():
            if abs(offset) > 50:  # More than 50 ticks is unusual
                issues.append(f"Large pocket offset for {inst}: {offset}")
    
    return issues
```

---

## 10. MISSING: Error Context
**File:** Throughout codebase

### Current Problem:
```python
raise FileNotFoundError("File not found")  # WHICH file??
raise ValueError("Invalid template")  # WHAT'S invalid??
```

### Fix Required:
```python
# Custom exceptions with context
class MusicBrainError(Exception):
    """Base exception for Music Brain."""
    pass

class TemplateError(MusicBrainError):
    """Template-related errors."""
    def __init__(self, template_name, issues):
        self.template_name = template_name
        self.issues = issues
        super().__init__(f"Template '{template_name}' validation failed: {'; '.join(issues)}")

class AudioAnalysisError(MusicBrainError):
    """Audio analysis errors."""
    def __init__(self, filepath, reason):
        self.filepath = filepath
        self.reason = reason
        super().__init__(f"Cannot analyze {filepath}: {reason}")

# Usage:
if issues:
    raise TemplateError(genre_name, issues)

if not filepath.exists():
    raise AudioAnalysisError(filepath, f"File does not exist: {filepath}")
```

---

## Summary of Required Fixes

1. **Immediate (Breaks functionality):**
   - Fix chord_to_degree ambiguous return
   - Fix progression matching logic
   - Add division-by-zero checks
   - Handle librosa API changes

2. **Important (Incorrect results):**
   - Add minor scale support
   - Fix duplicate patterns
   - Add key parsing validation
   - Implement per-instrument velocity

3. **Quality (Better errors):**
   - Add template validation
   - Add custom exceptions
   - Add logging instead of print
   - Add input sanitization

Each fix is estimated at 1-4 hours of development time, with testing doubling that estimate.
