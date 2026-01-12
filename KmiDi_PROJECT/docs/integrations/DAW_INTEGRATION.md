# DAW Integration Guide

This guide explains how to integrate DAiW Music-Brain with popular Digital Audio Workstations (DAWs).

## Table of Contents

- [Logic Pro X/Pro](#logic-pro-xpro)
- [Ableton Live](#ableton-live)
- [FL Studio](#fl-studio)
- [Pro Tools](#pro-tools)
- [Cubase/Nuendo](#cubasenuendo)
- [Studio One](#studio-one)
- [Reaper](#reaper)
- [Bitwig Studio](#bitwig-studio)

---

## Logic Pro X/Pro

### Integration Methods

#### 1. MIDI File Export/Import
The simplest method - generate MIDI files with DAiW and import into Logic.

```bash
# Generate chord progression
daiw intent process my_song_intent.json -o output.mid

# Import into Logic:
# File → Import → MIDI File
```

#### 2. OSC (Open Sound Control)
Real-time communication between DAiW and Logic Pro.

**Setup:**

1. Enable OSC in Logic Pro:
   - Logic Pro → Settings → MIDI → OSC
   - Enable "OSC Communication"
   - Note the port number (default: 9000)

2. Configure DAiW for OSC:
```python
from music_brain.integration import LogicProOSC

# Connect to Logic
logic = LogicProOSC(host='localhost', port=9000)

# Send chord progression
logic.send_progression(['Cmaj7', 'Fmaj7', 'Dm7', 'G7'])

# Control transport
logic.play()
logic.stop()
logic.set_tempo(120)
```

#### 3. Logic Pro Scripting
Use Logic's JavaScript automation (Logic Pro 10.5+).

**Example Script** (`logic_scripts/import_daiw.js`):
```javascript
// Import DAiW-generated MIDI
var Scripter = PluginParameters[0];

function HandleMIDI(event) {
    // Process MIDI from DAiW
    event.send();
}

function ProcessMIDI() {
    // Handle timing
}
```

**Load in Logic:**
1. Add MIDI FX → Scripter
2. Load script: `import_daiw.js`
3. Configure parameters

#### 4. Apple Script Automation

```applescript
-- DAiW Logic Pro Integration
-- Save as: daiw_logic_import.scpt

tell application "Logic Pro"
    activate
    
    -- Import MIDI file
    set midiFile to POSIX file "/path/to/output.mid"
    open midiFile
    
    -- Set tempo
    set tempo to 120
    
    -- Start playback
    play
end tell
```

Run from DAiW:
```python
import subprocess
subprocess.run(['osascript', 'daiw_logic_import.scpt'])
```

### Workflow Examples

**Example 1: Generate Drums with Lo-Fi Feel**
```bash
# Create intent file
daiw intent new --title "Lo-Fi Drums" --emotion grief

# Edit intent.json to specify:
# - mood_primary: "grief"
# - technical_genre: "lofi"
# - groove: "behind the beat"

# Process and generate
daiw intent process intent.json -o drums.mid

# Import into Logic Pro
# Apply: Ultrabeat, RC-20, vinyl emulation
```

**Example 2: Chord Progression with Modal Interchange**
```bash
# Diagnose existing progression
daiw diagnose "F-C-Am-Dm" --key "D minor"

# Get reharmonization suggestions
daiw reharm "F-C-Am-Dm" --style emotional

# Process with intent system for full arrangement
daiw intent process kelly_song.json -o arrangement.mid
```

### Logic Pro Templates

Pre-configured Logic templates for DAiW workflows available in `templates/logic_pro/`:

- `daiw_lofi_bedroom.logicx` - Lo-fi bedroom pop template
- `daiw_emotional_indie.logicx` - Emotional indie template
- `daiw_grief_folk.logicx` - Grief/folk acoustic template

---

## Ableton Live

### Integration Methods

#### 1. MIDI File Drag & Drop
```bash
# Generate MIDI
daiw intent process intent.json -o track.mid

# Drag output.mid into Ableton Live arrangement view
```

#### 2. Max for Live Device
Custom Max for Live device for real-time integration.

**Installation:**
1. Copy `max4live/DAiW_Control.amxd` to Ableton User Library
2. Add device to MIDI track
3. Configure port in device settings

**Features:**
- Real-time groove extraction
- Intent-based generation
- Chord progression suggestions
- Rule-breaking teaching mode

#### 3. Python Remote Script
Ableton Live 11+ with MIDI Remote Scripts.

**Setup:**
```bash
# Copy remote script
cp -r ableton_remote_scripts/DAiW_Remote \
      ~/Music/Ableton/User\ Library/Remote\ Scripts/

# Restart Ableton Live
# Preferences → Link/Tempo/MIDI → Control Surface → DAiW Remote
```

**Usage:**
```python
# From DAiW
from music_brain.integration import AbletonRemote

ableton = AbletonRemote()
ableton.set_tempo(120)
ableton.create_midi_track("DAiW Generated")
ableton.send_midi_clip(midi_data, track=0, clip_slot=0)
```

#### 4. MIDI Loopback
Use virtual MIDI ports for real-time communication.

**macOS:**
```bash
# Enable IAC Driver
# Audio MIDI Setup → MIDI Studio → IAC Driver → Device is online

# Send from DAiW
python -c "from music_brain.integration import send_midi_realtime; \
           send_midi_realtime('IAC Driver Bus 1', progression)"
```

**Windows:**
```powershell
# Install loopMIDI: https://www.tobias-erichsen.de/software/loopmidi.html
# Create port: "DAiW Bus"
```

### Workflow Examples

**Example 1: Lo-Fi Hip-Hop Beat**
```python
from music_brain.groove import GrooveApplicator
from music_brain.session import process_intent

# Extract groove from reference
extractor.extract_groove('reference_beat.mid', 'my_groove.json')

# Apply to quantized drums
applicator.apply_groove('drums.mid', 'output_drums.mid', 
                        groove='my_groove.json', intensity=0.8)

# Import output_drums.mid into Ableton
```

**Example 2: Generative Ambient**
```python
# Use intent system for ambient mood
intent = {
    "mood_primary": "nostalgia",
    "vulnerability_scale": "high",
    "technical_genre": "ambient",
    "technical_rule_to_break": "HARMONY_UnresolvedDissonance"
}

# Generate evolving progression
processor.process_intent(intent, output='ambient_progression.mid')
```

### Ableton Templates

Templates in `templates/ableton/`:

- `DAiW_Lofi_Starter.als` - Lo-fi production
- `DAiW_Indie_Electronic.als` - Indie electronic
- `DAiW_Ambient_Pad.als` - Ambient/atmospheric

---

## FL Studio

### Integration Methods

#### 1. MIDI Import
```bash
daiw intent process intent.json -o melody.mid

# FL Studio: File → Import → MIDI file
```

#### 2. VST3 Plugin (Coming Soon)
Native FL Studio plugin for DAiW integration.

#### 3. MIDI Scripting
FL Studio's MIDI scripting API.

**Script:** `fl_scripts/daiw_import.py`
```python
import transport
import midi
import channels

def OnMidiIn(event):
    # Route DAiW MIDI to channels
    event.handled = True
    
def OnIdle():
    # Check for new data from DAiW
    pass
```

### Workflow Example

**Hip-Hop Drum Programming:**
```bash
# Extract groove from sample
daiw extract sample_drums.mid -o groove.json

# Create drums with extracted feel
daiw apply quantized_pattern.mid --groove groove.json

# Import into FL Studio Channel Rack
```

---

## Pro Tools

### Integration Methods

#### 1. MIDI File Import
```bash
daiw intent process intent.json -o session.mid

# Pro Tools: File → Import → MIDI
```

#### 2. AAX Plugin (Coming Soon)
Native AAX plugin for Pro Tools.

#### 3. ReWire (Legacy)
For older Pro Tools versions with ReWire support.

### Workflow Example

**Film Scoring:**
```python
# Generate emotional underscore
from music_brain.session import CompleteSongIntent

intent = CompleteSongIntent(
    song_root={
        "core_event": "Character's moment of realization"
    },
    song_intent={
        "mood_primary": "bittersweet",
        "narrative_arc": "Slow Reveal"
    },
    technical_constraints={
        "technical_key": "F",
        "technical_mode": "mixolydian"
    }
)

# Outputs orchestral MIDI for Pro Tools
```

---

## Cubase/Nuendo

### Integration Methods

#### 1. MIDI Import
Standard MIDI file import.

#### 2. VST3 Plugin (Planned)
Full integration via VST3.

#### 3. Generic Remote
MIDI control surface for DAiW parameters.

**Setup:**
1. Devices → Device Setup → Generic Remote
2. Import `cubase_remote/DAiW_Remote.xml`

---

## Studio One

### Integration Methods

#### 1. Drag & Drop MIDI
Generate and drag MIDI files.

#### 2. VST3 Plugin (Planned)
Native Studio One integration.

#### 3. Macro Control
Use macros for automated workflows.

---

## Reaper

### Integration Methods

#### 1. MIDI Import
```bash
daiw intent process intent.json -o track.mid
```

#### 2. ReaScript
Custom ReaScript for automation.

**Script:** `reaper_scripts/DAiW_Import.lua`
```lua
-- DAiW MIDI Import for Reaper
function import_daiw_midi(file)
    reaper.InsertMedia(file, 0)
end
```

#### 3. JSFX Plugin (Planned)
Custom JSFX effect for groove/harmony processing.

### Workflow Example

**Live Looping with Groove Extraction:**
```python
# Extract groove from live performance
extractor.extract_groove('live_recording.mid')

# Apply to loops in real-time
# Use Reaper's MIDI routing
```

---

## Bitwig Studio

### Integration Methods

#### 1. MIDI File Import
Standard MIDI import.

#### 2. Controller Script
Custom Bitwig controller script.

**Location:** `~/Documents/Bitwig Studio/Controller Scripts/DAiW/`

#### 3. VST3 Plugin (Planned)
Native Bitwig integration.

### Workflow Example

**Modular Integration:**
```python
# Generate modulation curves from emotional intent
from music_brain.data.emotional_mapping import get_parameters_for_state

params = get_parameters_for_state(EmotionalState(
    valence=-0.8,
    arousal=0.3
))

# Export as MIDI CC for Bitwig modulators
```

---

## General Tips

### Best Practices

1. **Use 24 PPQ MIDI**: DAiW works best with 24 or 480 PPQ resolution
2. **Preserve Humanization**: Don't quantize after applying groove
3. **Backup Original**: Keep quantized versions before applying DAiW processing
4. **Render in Sections**: Process verse, chorus, bridge separately for variation

### MIDI Routing

**macOS Virtual MIDI:**
```bash
# Audio MIDI Setup → MIDI Studio → IAC Driver
# Enable "Device is online"
```

**Windows Virtual MIDI:**
```powershell
# Install loopMIDI
# Create virtual ports for routing
```

**Linux Virtual MIDI:**
```bash
# Load ALSA loopback
sudo modprobe snd-virmidi

# List MIDI ports
aconnect -l
```

### Troubleshooting

**MIDI Not Importing:**
- Check MIDI format (Format 1 recommended)
- Verify tempo map inclusion
- Check track count limits

**Timing Issues:**
- Ensure PPQ matches (use DAiW's `--ppq` flag)
- Check DAW's timing resolution
- Verify groove intensity not too extreme

**Missing Notes:**
- Check velocity thresholds
- Verify note range compatibility
- Ensure no note length < 1 tick

---

## Next Steps

1. Choose your DAW integration method
2. Follow setup instructions
3. Try example workflows
4. Customize for your needs

For DAW-specific questions, see:

- [FAQ](FAQ.md)
- [Troubleshooting](TROUBLESHOOTING.md)
- [GitHub Discussions](https://github.com/yourusername/DAiW-Music-Brain/discussions)
