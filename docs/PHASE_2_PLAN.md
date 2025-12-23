# DAiW Phase 2: Audio Engine - Complete Implementation Plan

## 🎯 PHASE 2 OVERVIEW

**Goal:** Integrate audio analysis capabilities to understand existing recordings and generate complete arrangements.

**Duration:** 6-8 weeks (can be done in parallel with Phase 1 completion)

**Current Status:**
- Phase 1: 92% complete (CLI + core modules working)
- Phase 2: 0% complete (planning stage)

---

## 📊 WHAT PHASE 2 ADDS

### **Current Capabilities (Phase 1):**
- ✅ Intent → Harmony (MIDI)
- ✅ Groove extraction/application (MIDI)
- ✅ Chord analysis (MIDI)
- ✅ Rule-breaking database

### **New Capabilities (Phase 2):**
- 🎵 **Audio Analysis** - Analyze WAV/MP3 files
- 🎸 **Arrangement Generation** - Complete song structures
- 🎚️ **Production Analysis** - Frequency, dynamics, effects
- 🎛️ **Audio-to-MIDI** - Extract chords/melody from audio
- 🎼 **Complete Composition** - Harmony + Groove + Arrangement

---

## 🔧 PHASE 2 ARCHITECTURE

```
Phase 2 Architecture:
┌─────────────────────────────────────────────────────────┐
│                    AUDIO ENGINE                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Audio      │  │ Arrangement  │  │  Production  │ │
│  │  Analysis    │  │  Generator   │  │   Analysis   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│         │                  │                  │         │
│         └──────────────────┴──────────────────┘         │
│                           │                              │
│  ┌────────────────────────────────────────────────────┐ │
│  │           Integration Layer                        │ │
│  │  (Connects to Phase 1: Harmony + Groove)          │ │
│  └────────────────────────────────────────────────────┘ │
│                           │                              │
│  ┌────────────────────────────────────────────────────┐ │
│  │              MIDI + Audio Output                   │ │
│  │   Complete arrangement with production notes       │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## 📋 PHASE 2 MODULES

### **Module 1: Audio Analysis Engine**
Extract musical information from audio files.

**Features:**
1. **Frequency Analysis** (Your 8-band system)
   - Low sub (20-60 Hz)
   - Sub bass (60-120 Hz)
   - Bass (120-250 Hz)
   - Low mids (250-500 Hz)
   - Mids (500-2k Hz)
   - High mids (2k-4k Hz)
   - Presence (4k-8k Hz)
   - Air (8k-20k Hz)

2. **Chord Detection**
   - Extract chord progressions from audio
   - Confidence scoring
   - Beat-aligned segmentation

3. **Tempo & Beat Detection**
   - BPM estimation
   - Beat grid extraction
   - Timing analysis

4. **Spectral Features**
   - Timbre analysis
   - Harmonic content
   - Genre fingerprinting

5. **Dynamics Analysis**
   - RMS levels
   - Peak detection
   - Dynamic range

**Dependencies:**
- `librosa` - Audio analysis
- `numpy` - Numerical processing
- `scipy` - Signal processing
- `aubio` - Pitch/beat detection

**Estimated Time:** 2 weeks

---

### **Module 2: Arrangement Generator**
Create complete song structures from intent.

**Features:**
1. **Section Generation**
   - Intro/Outro
   - Verse/Chorus/Bridge
   - Pre-chorus/Post-chorus
   - Breakdown/Build

2. **Arrangement Templates**
   - Genre-specific structures
   - Verse-Chorus-Verse-Chorus-Bridge-Chorus
   - AABA, ABAB, AAA patterns
   - Custom templates

3. **Dynamic Arc**
   - Energy curve over time
   - Tension/release mapping
   - Emotional journey

4. **Instrumentation Planning**
   - Which instruments per section
   - Layering suggestions
   - Drop-outs for dynamics

5. **Production Notes**
   - Where to break rules
   - Effect suggestions
   - Mix notes

**Integration:**
- Uses Phase 1 harmony generator
- Uses Phase 1 groove system
- Adds arrangement structure

**Estimated Time:** 2-3 weeks

---

### **Module 3: Production Analysis**
Analyze production techniques in reference tracks.

**Features:**
1. **Frequency Balance**
   - 8-band spectral analysis
   - Genre-specific profiles
   - Mix recommendations

2. **Stereo Field Analysis**
   - Width measurement
   - Pan distribution
   - Mono compatibility

3. **Effects Detection**
   - Reverb estimation
   - Delay patterns
   - Saturation/distortion

4. **Reference Matching**
   - Compare your mix to reference
   - Frequency response matching
   - Dynamic range comparison

5. **Production Fingerprinting**
   - "Sounds like" analysis
   - Era classification
   - Genre markers

**Use Cases:**
- Analyze Kelly song references (Elliott Smith, Bon Iver)
- Extract lo-fi production characteristics
- Generate production guidelines

**Estimated Time:** 2 weeks

---

### **Module 4: Complete Composition Pipeline**
Wire everything together into one workflow.

**Features:**
1. **Full Song Generation**
   ```python
   song = complete_song_from_intent(
       intent=kelly_intent,
       reference_audio="elliott_smith_reference.wav",
       output_format="midi+arrangement"
   )
   ```

2. **Multi-track MIDI Output**
   - Track 1: Chords (from harmony generator)
   - Track 2: Bass (generated from chords)
   - Track 3: Drums (with groove applied)
   - Track 4: Melody (optional)
   - Track 5: Arrangement markers

3. **Production Document**
   - Mix notes per section
   - Effect suggestions
   - Reference comparisons
   - Rule-breaking justifications

4. **DAW Project Template**
   - Pre-configured Logic/Ableton project
   - MIDI imported
   - Markers set
   - Notes embedded

**Integration Point:**
This is where everything comes together.

**Estimated Time:** 1-2 weeks

---

## 🎯 PHASE 2 PRIORITIES

### **Priority 1: Audio Analysis Core (2 weeks)**
**Why First:** Foundation for everything else

**Tasks:**
1. Set up `librosa` integration
2. Implement 8-band frequency analysis
3. Build chord detection (basic)
4. Create tempo/beat detection
5. Test with Kelly song references

**Deliverable:**
```python
audio_analyzer = AudioAnalyzer()
analysis = audio_analyzer.analyze_file("reference.wav")

# Output:
# {
#   'tempo': 82.0,
#   'key': 'F',
#   'chords': ['F', 'C', 'Am', 'Dm'],
#   'frequency_profile': {8-band data},
#   'dynamic_range': 12.5  # dB
# }
```

---

### **Priority 2: Arrangement Generator (2-3 weeks)**
**Why Second:** Can work with Phase 1 MIDI while audio analysis develops

**Tasks:**
1. Define arrangement data structures
2. Create section templates (verse, chorus, etc.)
3. Build energy arc calculator
4. Implement genre-specific arrangements
5. Add instrumentation planning

**Deliverable:**
```python
arranger = ArrangementGenerator()
arrangement = arranger.generate_from_intent(
    intent=kelly_intent,
    structure="verse-chorus-verse-chorus-bridge-chorus"
)

# Output:
# Section map with timing
# Instrumentation per section
# Dynamic arc curve
# Production notes
```

---

### **Priority 3: Integration & Testing (1-2 weeks)**
**Why Third:** Bring audio + arrangement together

**Tasks:**
1. Wire audio analysis → arrangement suggestions
2. Create complete composition pipeline
3. Test with Kelly song workflow
4. Generate full MIDI arrangements
5. Create production documents

**Deliverable:**
Complete Kelly song MIDI package with:
- Harmony (chords)
- Drums (with groove)
- Bass line
- Arrangement markers
- Production notes

---

### **Priority 4: Production Analysis (2 weeks)**
**Why Last:** Advanced feature, not blocking

**Tasks:**
1. Implement stereo field analysis
2. Build reference matching
3. Create production fingerprinting
4. Add mix recommendations
5. Genre classification

**Deliverable:**
```python
prod_analyzer = ProductionAnalyzer()
analysis = prod_analyzer.analyze_reference("elliott_smith_waltz2.wav")

# Output:
# Lo-fi characteristics detected
# Frequency profile
# Recommended production techniques
# "Sounds like" matches
```

---

## 🎼 KELLY SONG - PHASE 2 WORKFLOW

### **Complete Kelly Song Generation:**

```python
from music_brain.audio import AudioAnalyzer
from music_brain.arrangement import ArrangementGenerator
from music_brain.composition import CompleteComposer

# 1. Analyze reference tracks
analyzer = AudioAnalyzer()
elliott_smith = analyzer.analyze_file("either_or_reference.wav")
bon_iver = analyzer.analyze_file("for_emma_reference.wav")

# 2. Extract production characteristics
lofi_profile = analyzer.extract_production_profile([
    elliott_smith, bon_iver
])

# Output:
# {
#   'frequency_balance': {8-band profile},
#   'dynamic_range': 'high (12-18 dB)',
#   'stereo_width': 'narrow (intimate)',
#   'effects': ['room_reverb', 'tape_saturation'],
#   'genre_markers': ['lo-fi', 'bedroom', 'confessional']
# }

# 3. Generate complete arrangement
composer = CompleteComposer()
kelly_song = composer.compose_from_intent(
    intent=kelly_intent,
    reference_profile=lofi_profile,
    structure="verse-verse-chorus-verse-chorus-bridge-chorus"
)

# Output:
# - kelly_song_complete.mid (multi-track MIDI)
# - kelly_song_arrangement.json (section markers)
# - kelly_song_production_notes.md (mix guide)

# 4. Import into Logic Pro X
# - Pre-configured project template
# - All MIDI imported
# - Markers set
# - Production notes as comments
```

---

## 📁 PHASE 2 FILE STRUCTURE

```
DAiW-Music-Brain/
├── music_brain/
│   ├── audio/                    # NEW - Phase 2
│   │   ├── __init__.py
│   │   ├── analyzer.py           # Core audio analysis
│   │   ├── chord_detection.py   # Extract chords from audio
│   │   ├── frequency.py          # 8-band analysis
│   │   ├── tempo_beat.py         # BPM & beat detection
│   │   └── production.py         # Production analysis
│   │
│   ├── arrangement/              # NEW - Phase 2
│   │   ├── __init__.py
│   │   ├── generator.py          # Arrangement generation
│   │   ├── templates.py          # Genre templates
│   │   ├── sections.py           # Verse/chorus/bridge
│   │   └── dynamics.py           # Energy arc calculator
│   │
│   ├── composition/              # NEW - Phase 2
│   │   ├── __init__.py
│   │   ├── complete.py           # Complete composition pipeline
│   │   ├── multi_track.py        # Multi-track MIDI generation
│   │   └── daw_export.py         # DAW project templates
│   │
│   ├── harmony/                  # Phase 1 ✅
│   ├── groove/                   # Phase 1 ✅
│   ├── structure/                # Phase 1 ✅
│   └── session/                  # Phase 1 ✅
│
└── vault/
    └── Production_Guides/        # NEW - Phase 2
        ├── lofi_aesthetic.md
        ├── frequency_reference.md
        └── genre_production.md
```

---

## 🔬 TECHNICAL SPECIFICATIONS

### **Audio Analysis Module Specs:**

```python
class AudioAnalyzer:
    """
    Analyze audio files for musical and production characteristics.
    """
    
    def analyze_file(
        self,
        audio_path: str,
        sample_rate: int = 22050
    ) -> AudioAnalysis:
        """
        Complete audio analysis.
        
        Returns:
            AudioAnalysis with:
            - tempo_bpm: float
            - key: str
            - chords: List[str]
            - beats: List[float]  # Beat times in seconds
            - frequency_profile: FrequencyProfile
            - dynamic_range: float
            - spectral_features: dict
        """
        pass
    
    def extract_chords(
        self,
        audio_path: str,
        beat_aligned: bool = True
    ) -> List[Tuple[float, str]]:
        """
        Extract chord progression with timestamps.
        
        Returns:
            List of (time, chord_name) tuples
        """
        pass
    
    def analyze_frequency_balance(
        self,
        audio_path: str
    ) -> FrequencyProfile:
        """
        8-band frequency analysis.
        
        Returns:
            FrequencyProfile with RMS levels per band
        """
        pass
```

### **Arrangement Generator Specs:**

```python
class ArrangementGenerator:
    """
    Generate song arrangements from intent.
    """
    
    def generate_from_intent(
        self,
        intent: CompleteSongIntent,
        structure: str = "verse-chorus-verse-chorus-bridge-chorus",
        target_duration: float = 180.0  # 3 minutes
    ) -> Arrangement:
        """
        Generate complete arrangement.
        
        Returns:
            Arrangement with:
            - sections: List[Section]  # Each section with timing
            - energy_curve: List[float]  # 0-1 per second
            - instrumentation: Dict[str, List[str]]  # Section -> instruments
            - production_notes: Dict[str, str]  # Section -> notes
        """
        pass
    
    def create_section(
        self,
        section_type: str,  # 'verse', 'chorus', 'bridge', etc.
        duration: float,
        energy_level: float = 0.5
    ) -> Section:
        """Create a single section with characteristics."""
        pass
```

---

## 🚀 PHASE 2 IMPLEMENTATION TIMELINE

### **Week 1-2: Audio Analysis Core**
- ✅ Set up librosa
- ✅ 8-band frequency analysis
- ✅ Basic chord detection
- ✅ Tempo/beat detection
- ✅ Test with references

### **Week 3-4: Arrangement Generator**
- ✅ Section templates
- ✅ Energy arc calculator
- ✅ Instrumentation planning
- ✅ Genre arrangements
- ✅ Integration with Phase 1

### **Week 5-6: Complete Composition**
- ✅ Multi-track MIDI generation
- ✅ Bass line generator
- ✅ Arrangement markers
- ✅ Production documents
- ✅ Kelly song complete workflow

### **Week 7-8: Production Analysis & Polish**
- ✅ Stereo field analysis
- ✅ Reference matching
- ✅ Production fingerprinting
- ✅ Documentation
- ✅ Testing & refinement

---

## 🎯 SUCCESS CRITERIA

**Phase 2 is complete when:**

1. ✅ Audio files can be analyzed for:
   - Tempo, key, chords
   - Frequency balance (8-band)
   - Dynamic characteristics

2. ✅ Complete arrangements can be generated:
   - Multi-section structures
   - Energy arcs
   - Instrumentation per section

3. ✅ Kelly song workflow produces:
   - Multi-track MIDI (chords, bass, drums)
   - Arrangement markers
   - Production notes document

4. ✅ Reference tracks can be analyzed:
   - "Sounds like" matching
   - Production characteristics
   - Mix recommendations

5. ✅ Everything integrates with Phase 1:
   - Uses harmony generator
   - Uses groove system
   - CLI commands work

---

## 💡 KELLY SONG - PHASE 2 SPECIFIC GOALS

### **Input:**
```
Kelly intent + Elliott Smith/Bon Iver references
```

### **Output:**
```
kelly_song_complete/
├── kelly_harmony.mid          # F-C-Dm-Bbm (Phase 1 ✅)
├── kelly_drums.mid            # With groove applied (Phase 1 ✅)
├── kelly_bass.mid             # NEW - Phase 2
├── kelly_arrangement.json     # NEW - Section timing
└── kelly_production_guide.md  # NEW - Mix notes
```

### **Production Guide Contents:**
```markdown
# Kelly Song - Production Guide

## Reference Analysis
Based on Elliott Smith "Either/Or" and Bon Iver "For Emma":

### Frequency Balance
- Low end: Minimal (intimate, bedroom)
- Mids: Present but not aggressive
- High end: Rolled off above 12kHz (warmth)

### Dynamics
- High dynamic range (12-18 dB)
- No compression on vocals (raw, intimate)
- Room noise present (authenticity marker)

### Effects
- Short room reverb (3-8ft space)
- Minimal delay
- Tape saturation on master (subtle)

### Stereo Field
- Narrow (mono-ish)
- Vocals centered
- Guitar slightly wide
- Drums mostly centered

## Section-by-Section Notes

### Verse 1 (0:00-0:45)
- Just guitar + vocals
- No drums yet (build anticipation)
- Minimal processing
- Let room noise live

### Verse 2 (0:45-1:30)
- Add subtle drums (60-70% velocity)
- Keep guitar as foundation
- Vocals still dry

### Chorus (1:30-2:00)
- Drums at full intensity (still not loud)
- Optional second guitar layer
- Slight reverb swell on "found you"

[... etc for all sections]

## Rule-Breaking Applied
- Harmonic: Bbm in F major (modal interchange)
- Production: Pitch imperfections left in
- Timing: Minimal groove (intimacy)
- Mix: Not radio-ready (authenticity)
```

---

## 🔧 TOOLS & LIBRARIES

### **Core Audio:**
```bash
pip install librosa --break-system-packages      # Audio analysis
pip install aubio --break-system-packages        # Pitch/beat detection
pip install numpy scipy --break-system-packages  # Numerical processing
```

### **Optional (Advanced):**
```bash
pip install essentia --break-system-packages     # Advanced audio analysis
pip install madmom --break-system-packages       # Beat tracking
pip install pydub --break-system-packages        # Audio manipulation
```

### **Already Have:**
```python
mido      # MIDI I/O (Phase 1)
```

---

## 📊 PHASE 2 VS PHASE 1

### **Phase 1 Scope:**
- Input: Intent (emotional/technical)
- Processing: MIDI generation/analysis
- Output: MIDI files
- **Result:** Individual musical elements

### **Phase 2 Scope:**
- Input: Intent + Audio references
- Processing: Audio analysis + arrangement
- Output: Complete multi-track MIDI + production notes
- **Result:** Complete song blueprint

### **The Difference:**
Phase 1 gives you **parts**.
Phase 2 gives you the **whole song**.

---

## 🎯 IMMEDIATE NEXT STEPS (After Phase 1 Complete)

### **Step 1: Set Up Audio Environment (30 min)**
```bash
# Install audio libraries
pip install librosa aubio numpy scipy --break-system-packages

# Test installation
python -c "import librosa; print('✓ librosa ready')"
python -c "import aubio; print('✓ aubio ready')"
```

### **Step 2: Create Audio Module Skeleton (1 hour)**
```bash
# Create directory structure
mkdir -p DAiW-Music-Brain/music_brain/audio
touch DAiW-Music-Brain/music_brain/audio/__init__.py
touch DAiW-Music-Brain/music_brain/audio/analyzer.py
```

### **Step 3: Implement Basic Audio Analysis (3-4 hours)**
Start with tempo/beat detection:
```python
import librosa

def analyze_audio_basic(audio_path):
    # Load audio
    y, sr = librosa.load(audio_path)
    
    # Estimate tempo
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    
    # Estimate key/chords (basic)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    
    return {
        'tempo': tempo,
        'beats': beats,
        'chroma': chroma
    }
```

### **Step 4: Test with Kelly References (1 hour)**
```python
# Analyze Elliott Smith track
analysis = analyze_audio_basic("elliott_smith_reference.wav")
print(f"Detected tempo: {analysis['tempo']} BPM")
print(f"Beat count: {len(analysis['beats'])}")

# Compare to Kelly song target (82 BPM)
tempo_match = abs(analysis['tempo'] - 82.0) < 10
print(f"Tempo matches Kelly: {tempo_match}")
```

---

## 📝 DOCUMENTATION TO CREATE

### **Phase 2 Guides:**
1. **Audio Analysis Guide**
   - How to analyze reference tracks
   - Interpreting frequency profiles
   - Extracting production characteristics

2. **Arrangement Guide**
   - Song structure templates
   - Energy arc design
   - Instrumentation planning

3. **Complete Composition Workflow**
   - End-to-end example
   - Kelly song walkthrough
   - Troubleshooting

4. **Production Analysis Guide**
   - Reference matching
   - Genre fingerprinting
   - Mix recommendations

---

## ✅ PHASE 2 CHECKLIST

**Audio Analysis:**
- [ ] Librosa integration
- [ ] 8-band frequency analysis
- [ ] Chord detection
- [ ] Tempo/beat detection
- [ ] Reference analysis working

**Arrangement Generator:**
- [ ] Section templates
- [ ] Energy arc calculator
- [ ] Instrumentation planner
- [ ] Genre-specific arrangements
- [ ] Integration with Phase 1

**Complete Composition:**
- [ ] Multi-track MIDI generation
- [ ] Bass line generator
- [ ] Arrangement markers
- [ ] Production documents
- [ ] Kelly song complete workflow

**Production Analysis:**
- [ ] Frequency matching
- [ ] Stereo field analysis
- [ ] Effects detection
- [ ] Reference comparison
- [ ] Genre classification

**Integration & Testing:**
- [ ] CLI commands
- [ ] Test suite
- [ ] Documentation
- [ ] Kelly song proof-of-concept
- [ ] Ready for Phase 3

---

## 🚀 PHASE 3 PREVIEW

After Phase 2 completes, Phase 3 will add:
- **Desktop GUI** (Ableton-style interface)
- **Visual arrangement editor**
- **Interactive MIDI preview**
- **Real-time audio playback**
- **Project management**

But that's 6-8 weeks away. Focus on Phase 2 first.

---

## 💬 PHILOSOPHY ALIGNMENT

**Phase 2 maintains "Interrogate Before Generate":**

1. **Audio Analysis = Understanding**
   - Not "copy this reference"
   - But "understand why this works"

2. **Arrangement = Emotional Journey**
   - Not "standard song structure"
   - But "structure that serves the emotion"

3. **Production Analysis = Authenticity**
   - Not "industry standard mix"
   - But "what production choices serve the song?"

4. **Complete Composition = Holistic**
   - Everything connects to intent
   - Every choice justified
   - No random decisions

---

## 🎵 THE ULTIMATE GOAL

**By end of Phase 2, you can:**

```python
# Complete Kelly song generation
kelly_song = generate_complete_song(
    intent=CompleteSongIntent(
        core_wound="Finding someone after they left",
        emotional_intent="Grief disguised as love",
        technical_constraints="F major, 82 BPM, lo-fi"
    ),
    references=[
        "elliott_smith_either_or.wav",
        "bon_iver_for_emma.wav"
    ],
    structure="intimate buildup"
)

# Result:
# - Complete multi-track MIDI
# - Arrangement with markers
# - Production guide
# - Ready to record over
```

**Input:** Emotional truth
**Output:** Complete song blueprint
**Philosophy:** "Interrogate Before Generate" at scale

---

## 📊 FINAL PHASE 2 SUMMARY

**Scope:** Audio analysis + Complete arrangements
**Duration:** 6-8 weeks
**Complexity:** Medium (builds on Phase 1)
**Dependencies:** Librosa, aubio, numpy, scipy

**Delivers:**
- Audio reference analysis
- Complete song arrangements
- Multi-track MIDI generation
- Production documentation
- Kelly song complete package

**Current Status:** Phase 1 at 92%, ready to start Phase 2
**Next Action:** Install audio libraries and create module skeleton

---

*"From intent to complete song - Phase 2 makes it real."*

Ready to start Phase 2 implementation? 🚀
