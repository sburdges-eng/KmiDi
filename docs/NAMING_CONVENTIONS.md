# Naming Conventions

**Date:** 2026-01-23  
**Project:** KmiDi-1  
**Phase:** 2.1.2 - Code Organization

## Overview

This document defines the naming conventions used throughout the KmiDi-1 codebase. Consistent naming improves code readability, maintainability, and reduces cognitive load when working with the codebase.

## Python Naming Conventions

### Classes

**Rule:** Use PascalCase (also known as CapWords)

**Examples:**
- `BassEngine`
- `IntentProcessor`
- `EmotionThesaurus`
- `CompleteSongIntent`
- `GrooveEngine`

**Rationale:** Classes represent types/blueprints and should be easily distinguishable from instances.

### Functions and Methods

**Rule:** Use snake_case

**Examples:**
- `process_intent()`
- `generate_melody()`
- `analyze_chord()`
- `get_emotion_mapping()`
- `create_session()`

**Rationale:** Functions represent actions and snake_case is the Python standard (PEP 8).

### Variables and Parameters

**Rule:** Use snake_case

**Examples:**
- `intent_data`
- `emotion_node`
- `tempo_range`
- `technical_constraints`
- `song_root`

**Rationale:** Consistent with Python conventions and function naming.

### Constants

**Rule:** Use UPPER_SNAKE_CASE

**Examples:**
- `MAX_TEMPO = 200`
- `DEFAULT_KEY = "C"`
- `MIN_INTENSITY = 0.0`
- `MAX_INTENSITY = 1.0`
- `EMOTION_CATEGORIES = [...]`

**Rationale:** Constants should be immediately recognizable and distinct from variables.

### Modules and Packages

**Rule:** Use snake_case

**Examples:**
- `intent_processor.py`
- `emotion_thesaurus.py`
- `groove_engine.py`
- `bass_engine.py`
- `kelly_companion/`

**Rationale:** Module names should be lowercase with underscores (PEP 8).

### Private Attributes and Methods

**Rule:** Prefix with single underscore `_` for internal use

**Examples:**
- `_parse_intent()`
- `_calculate_harmony()`
- `_internal_state`
- `_cache_data`

**Rationale:** Indicates that these are implementation details and not part of the public API.

### Protected Attributes and Methods

**Rule:** Prefix with single underscore `_` (Python convention - no true "protected" access)

**Examples:**
- `_on_join` (callback)
- `_session_manager` (internal manager)

**Rationale:** Python doesn't have true protected access, but underscore prefix signals "internal use."

### Magic/Dunder Methods

**Rule:** Use double underscores `__method__`

**Examples:**
- `__init__()`
- `__repr__()`
- `__str__()`
- `__len__()`

**Rationale:** Python convention for special methods.

## C++ Naming Conventions

### Classes and Structs

**Rule:** Use PascalCase

**Examples:**
- `OSCMessage`
- `AudioAnalyzer`
- `GrooveEngine`
- `HarmonySystem`

### Functions and Methods

**Rule:** Use camelCase

**Examples:**
- `analyzeChord()`
- `generateMelody()`
- `processAudio()`
- `sendMessage()`

### Variables

**Rule:** Use camelCase

**Examples:**
- `tempoRange`
- `intentData`
- `emotionNode`

### Constants

**Rule:** Use UPPER_SNAKE_CASE

**Examples:**
- `MAX_TEMPO`
- `DEFAULT_KEY`
- `SAMPLE_RATE`

### Private Members

**Rule:** Prefix with underscore `_` or use `m_` prefix

**Examples:**
- `_internalState` or `m_internalState`
- `_cacheData` or `m_cacheData`

## File Naming

### Python Files

**Rule:** Use snake_case with `.py` extension

**Examples:**
- `intent_processor.py`
- `emotion_thesaurus.py`
- `bass_engine.py`

### C++ Files

**Rule:** Use PascalCase for headers, camelCase for sources

**Examples:**
- Headers: `OSCMessage.h`, `AudioAnalyzer.h`
- Sources: `oscMessage.cpp`, `audioAnalyzer.cpp`

**Note:** Current codebase uses mixed conventions - standardize going forward.

## Directory Naming

**Rule:** Use snake_case

**Examples:**
- `kelly_companion/`
- `groove_engine/`
- `intent_processor/`

## Abbreviations

**Rule:** Avoid abbreviations unless they are widely understood

**Good:**
- `intent` (not `int`)
- `processor` (not `proc`)
- `generator` (not `gen`)

**Acceptable:**
- `API` (Application Programming Interface)
- `OSC` (Open Sound Control)
- `MIDI` (Musical Instrument Digital Interface)
- `CLI` (Command Line Interface)

## Naming Patterns

### Boolean Variables

**Rule:** Use `is_`, `has_`, `can_`, or `should_` prefix

**Examples:**
- `is_active`
- `has_emotion`
- `can_edit`
- `should_break_rule`

### Collections

**Rule:** Use plural nouns

**Examples:**
- `emotions` (list of emotion objects)
- `engines` (list of engine instances)
- `participants` (list of session participants)

### Getters and Setters

**Rule:** Use descriptive names, avoid `get_`/`set_` prefix unless needed for clarity

**Examples:**
- `emotion()` (getter) and `emotion(value)` (setter) - Python properties
- `getAddress()` (C++ getter) and `setAddress()` (C++ setter)

## Migration Notes

### Current Inconsistencies

1. **C++ files:** Mixed naming (some PascalCase, some camelCase)
2. **Some modules:** Mixed conventions in older code
3. **Legacy code:** May not follow these conventions

### Migration Strategy

1. **New code:** Must follow these conventions
2. **Refactored code:** Update to follow conventions
3. **Legacy code:** Document exceptions, update when touched

## Examples

### Good Python Example

```python
class BassEngine:
    """Generates bass lines from musical intent."""
    
    MAX_VELOCITY = 127
    DEFAULT_OCTAVE = 3
    
    def __init__(self, intent: CompleteSongIntent):
        self._intent = intent
        self._cache = {}
    
    def generate_bass_line(self, length: int) -> List[MIDINote]:
        """Generate a bass line of specified length."""
        if length in self._cache:
            return self._cache[length]
        
        notes = self._calculate_notes(length)
        self._cache[length] = notes
        return notes
    
    def _calculate_notes(self, length: int) -> List[MIDINote]:
        """Internal method to calculate notes."""
        # Implementation...
        pass
```

### Good C++ Example

```cpp
class OSCMessage {
public:
    static constexpr int MAX_ARGUMENTS = 32;
    
    void setAddress(const std::string& address);
    const std::string& getAddress() const;
    
private:
    std::string m_address;
    std::vector<OSCArgument> m_arguments;
};
```

## References

- [PEP 8 - Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- [JUCE Coding Standards](https://juce.com/learn/coding-standards)

## Enforcement

- **Linters:** Configure flake8/pylint for Python, clang-format for C++
- **Code Reviews:** Check naming conventions during review
- **Documentation:** Update this guide as conventions evolve
