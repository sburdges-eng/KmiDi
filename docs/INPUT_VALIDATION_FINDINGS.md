# Input Validation Security Review - Findings

## Executive Summary

This security review identified **10 critical input validation issues** across `music_brain/api.py` and `music_brain/session/intent_schema.py` that could lead to crashes, silent failures, or incorrect behavior when processing user input from JSON, UI, or LLMs.

## Issues Found

### CRITICAL - Crashes from Missing Keys

#### Issue #1: Dictionary Access Without `.get()` in intent_schema.py
**Lines:** 427, 437, 447, 460
**Risk:** **CRASH** - KeyError when required keys missing from from_dict input

```python
# Line 427
root = data["song_root"]  # Crashes if key missing
# Line 437
si = data["song_intent"]  # Crashes if key missing
# Line 447
tc = data["technical_constraints"]  # Crashes if key missing
# Line 460
sd = data["system_directive"]  # Crashes if key missing
```

**Example Crash:**
```python
intent = CompleteSongIntent.from_dict({"title": "Test"})  # Missing song_root
# KeyError: 'song_root'
```

**Fix:** Use `.get()` with empty dict fallback


#### Issue #2: Dictionary Access Without `.get()` in api.py
**Lines:** 651-674, 1407
**Risk:** **CRASH** - KeyError when process_intent returns unexpected structure

```python
# Line 651-674
output = {
    "intent_summary": result['intent_summary'],  # Direct access
    "harmony": {
        "chords": result['harmony'].chords,  # Crashes if missing
    }
}

# Line 1407
response["audio_path"] = result["midi_path"].replace(...)  # Crashes if no midi_path
```

**Example Crash:**
```python
# If process_intent() returns incomplete result
result = {"other_key": "value"}  # Missing 'intent_summary'
# KeyError: 'intent_summary'
```

**Fix:** Use `.get()` with appropriate defaults


### HIGH - Type Coercion Without Validation

#### Issue #3: Float Conversion Without Bounds Checking
**File:** intent_schema.py
**Lines:** 353-360
**Risk:** **SILENT WRONG OUTPUT** - Accepts invalid range values

```python
try:
    tension_val = float(mood_secondary_tension)
except Exception:
    tension_val = 0.5
```

**Problem:** Accepts `999.5` or `-50` when spec says 0.0-1.0

**Example Wrong Behavior:**
```python
intent = CompleteSongIntent(mood_secondary_tension="999")
# tension_val = 999.0 (should reject or clamp)
```

**Fix:** Validate and clamp to [0.0, 1.0] range


#### Issue #4: Int Conversion Without Try-Except
**File:** api.py
**Line:** 1372
**Risk:** **CRASH** - ValueError when BPM is non-numeric

```python
motivation = max(1, min(10, int(request.intent.technical.bpm / 20)))
```

**Example Crash:**
```python
request.intent.technical.bpm = "fast"  # String instead of int
# TypeError: unsupported operand type(s) for /: 'str' and 'int'
```

**Fix:** Add try-except with default value


#### Issue #5: BPM Without Upper Bounds Check
**File:** api.py
**Lines:** 1216-1217
**Risk:** **SILENT WRONG OUTPUT** - Unrealistic tempo values

```python
bpm = tech.bpm if tech and tech.bpm is not None else 82
tempo_range = (max(60, bpm - 20), min(140, bpm + 20))
```

**Problem:** BPM=9999 creates tempo_range=(9979, 10019)

**Example Wrong Behavior:**
```python
request.intent.technical.bpm = 10000
# tempo_range = (9979, 10019) instead of rejecting or clamping
```

**Fix:** Validate BPM in reasonable range (40-300)


#### Issue #6: Duration Without Positive Validation
**File:** api.py
**Line:** 1259
**Risk:** **SILENT WRONG OUTPUT** / **CRASH** - Negative or zero bars

```python
duration_minutes = tech.duration if tech and tech.duration is not None else 3.0
length_bars = int((duration_minutes * bpm) / 4)  # Line 1259
```

**Problem:** duration=-5 creates negative bars

**Example Wrong Behavior:**
```python
request.intent.technical.duration = -5
# length_bars = negative number, downstream crashes or empty output
```

**Fix:** Validate duration > 0


### MEDIUM - Enum Validation Missing

#### Issue #7: Enum Values Used as Strings Without Validation
**File:** intent_schema.py
**Lines:** 440-443
**Risk:** **SILENT WRONG OUTPUT** - Invalid enum values accepted

```python
# Line 442
vulnerability_scale=si.get("vulnerability_scale", "Medium"),
```

**Problem:** Accepts any string. Line 540 compares `== "High"` but enum is VulnerabilityScale.HIGH

**Example Wrong Behavior:**
```python
data = {"song_intent": {"vulnerability_scale": "INVALID"}}
intent = CompleteSongIntent.from_dict(data)
# vulnerability_scale = "INVALID" (should be Low/Medium/High)
```

**Fix:** Validate against VulnerabilityScale enum values


#### Issue #8: Key Mode Parsing Without Validation
**File:** api.py
**Lines:** 772-775
**Risk:** **SILENT WRONG OUTPUT** - Invalid mode accepted

```python
key_parts = tech["key"].split()
technical_key = key_parts[0] if key_parts else "C"
if len(key_parts) > 1:
    technical_mode = key_parts[1].lower()  # No validation
```

**Problem:** Accepts "C xyz" → mode="xyz" (should be major/minor/dorian/etc)

**Example Wrong Behavior:**
```python
request.intent.technical.key = "F invalid_mode"
# technical_mode = "invalid_mode" instead of major/minor
```

**Fix:** Validate mode against known modes or default to "major"


### MEDIUM - Type Inconsistency

#### Issue #9: Vulnerability Scale Type Mismatch
**File:** intent_schema.py
**Lines:** 270, 331, 358-366
**Risk:** **SILENT WRONG OUTPUT** - Type confusion

**Problem:** 
- Line 270: `vulnerability_scale: str = "Medium"` (dataclass field is string)
- Line 331: `vulnerability_scale: float = 0.0` (parameter is float)
- Line 358-366: Converts to float then assigns to string field

**Example Wrong Behavior:**
```python
intent = CompleteSongIntent(vulnerability_scale=0.8)
# intent.song_intent.vulnerability_scale = 0.8 (float)
# But dataclass expects string "Low"/"Medium"/"High"
```

**Fix:** Standardize on either float OR enum string, not mix


### LOW - Optional Field Handling

#### Issue #10: Optional Fields Not Consistently Validated
**File:** api.py
**Lines:** Various
**Risk:** **SILENT FAILURE** - Missing validation before use

**Problem:** Optional fields like `tech.duration`, `tech.structure`, `tech.instruments` are checked for None but not validated for type/content

**Fix:** Add validation helpers for optional complex fields


## Summary by Category

### Crashes (CRITICAL)
- Issue #1: Dict access in from_dict (4 locations)
- Issue #2: Dict access in API (2 locations)
- Issue #4: Int conversion without try-except (1 location)

**Total:** 7 crash points

### Silent Wrong Output (HIGH)
- Issue #3: Float without bounds
- Issue #5: BPM without upper bounds
- Issue #6: Duration without positive check
- Issue #7: Enum string validation
- Issue #8: Mode validation
- Issue #9: Type mismatch

**Total:** 6 silent failure points

### Silent Fallback (MEDIUM)
- Issue #10: Optional field validation

**Total:** 1 silent fallback issue

## Recommended Fixes Priority

### P0 (Critical - Fix Immediately)
1. Add `.get()` to all dict access in from_dict
2. Add `.get()` to result dict access in API
3. Add try-except to int/float conversions

### P1 (High - Fix Soon)
4. Add bounds validation for tension (0-1)
5. Add bounds validation for BPM (40-300)
6. Add positive validation for duration
7. Add enum validation for vulnerability_scale

### P2 (Medium - Fix Next)
8. Add mode validation
9. Resolve vulnerability_scale type inconsistency

## Testing Recommendations

1. **Crash Tests:** Send malformed JSON missing required keys
2. **Bounds Tests:** Send extreme numeric values (0, -1, 99999)
3. **Type Tests:** Send strings where numbers expected
4. **Enum Tests:** Send invalid enum string values
5. **Empty Tests:** Send empty dicts/arrays

## No Issues Found In

- Emotion thesaurus lookup (has proper defaults)
- Production preset selection (has fallback logic)
- Groove template handling (validated internally)

---

**Review Date:** 2026-02-03
**Reviewer:** GitHub Copilot Security Agent
**Files Reviewed:** music_brain/api.py, music_brain/session/intent_schema.py
