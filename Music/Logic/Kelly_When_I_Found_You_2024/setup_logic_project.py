#!/usr/bin/env python3
"""
Logic Pro Project Setup for Kelly Song
Creates project structure and AppleScript for automation

Run this after generate_midi.py
"""

import os
import subprocess

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_NAME = "When_I_Found_You"

# AppleScript to create Logic Pro project with proper setup
LOGIC_SETUP_APPLESCRIPT = '''
tell application "Logic Pro"
    activate
    delay 1
end tell

tell application "System Events"
    tell process "Logic Pro"
        -- Create new project (Cmd+N)
        keystroke "n" using command down
        delay 2
        
        -- Press Enter to accept default template or empty project
        keystroke return
        delay 3
    end tell
end tell
'''

# Marker positions for the song (in bars, at 72 BPM)
MARKERS = [
    (1, "INTRO - Fingerpicked"),
    (9, "VERSE 1 - Light Strum"),
    (17, "VERSE 2 - Building"),
    (25, "CHORUS - Fuller"),
    (33, "INSTRUMENTAL - Arpeggiated/Diminished"),
    (37, "VERSE 3 - Pulled Back"),
    (45, "BRIDGE - Sustained"),
    (49, "FINAL - The Reveal"),
    (54, "OUTRO - Fade"),
]

def create_logic_template_instructions():
    """Create detailed instructions for Logic Pro setup"""
    
    instructions = """
================================================================================
LOGIC PRO PROJECT SETUP - "When I Found You" (Kelly Song)
================================================================================

STEP 1: CREATE NEW PROJECT
--------------------------
1. Open Logic Pro
2. File > New (or Cmd+N)
3. Choose "Empty Project"
4. Save as: "When_I_Found_You" in this folder

STEP 2: PROJECT SETTINGS
------------------------
1. File > Project Settings > Tempo
   - Set tempo to: 72 BPM
   
2. File > Project Settings > Key Signature
   - Set key to: A minor

3. File > Project Settings > Time Signature
   - Set to: 4/4

STEP 3: CREATE TRACKS
---------------------
Create the following tracks (Track > New Tracks or Option+Cmd+N):

TRACK 1: "Guitar L"
  - Type: Audio
  - Input: Your audio interface input 1 (or mono input)
  - Output: Stereo Out
  - Pan: -30 (left)
  - Add plugins: Light compression, subtle room reverb

TRACK 2: "Guitar R" 
  - Type: Audio
  - Input: Your audio interface input 2 (or same as Guitar L for double-tracking)
  - Output: Stereo Out
  - Pan: +30 (right)
  - Add plugins: Match Guitar L

TRACK 3: "Vocal"
  - Type: Audio
  - Input: Your vocal mic input
  - Output: Stereo Out
  - Pan: Center
  - Add plugins: Gentle compression, de-esser if needed, light reverb

TRACK 4: "MIDI Reference" (optional - delete after recording)
  - Type: Software Instrument
  - Load: Any piano or guitar sound
  - Import: kelly_song_reference.mid from this folder
  - Purpose: Guide track for timing, delete before mixing

STEP 4: SET UP MARKERS
----------------------
Add the following markers (Option+' to create marker):

Bar 1   - "INTRO - Fingerpicked"
Bar 9   - "VERSE 1 - Light Strum"  
Bar 17  - "VERSE 2 - Building"
Bar 25  - "CHORUS - Fuller"
Bar 33  - "INSTRUMENTAL - Arpeggiated"
Bar 37  - "VERSE 3 - Pulled Back"
Bar 45  - "BRIDGE - Sustained"
Bar 49  - "FINAL - The Reveal"
Bar 54  - "OUTRO - Fade"

STEP 5: SET UP CYCLE/LOOP REGIONS (for recording sections)
----------------------------------------------------------
You can use the cycle region (yellow bar at top) to loop sections while recording:
- INTRO: Bars 1-8
- VERSES: 8 bars each
- CHORUS: Bars 25-32
- etc.

STEP 6: IMPORT MIDI REFERENCE
-----------------------------
1. File > Import > MIDI File
2. Select: kelly_song_reference.mid
3. This creates a guide track showing chord changes and timing
4. Mute or delete this track after you've recorded your guitar parts

STEP 7: RECORDING TIPS
----------------------
- Record Guitar L first (full playthrough)
- Record Guitar R as a double (same part, slight variations add width)
- Record Vocal last, once guitar feels right
- Use punch-in recording (Autopunch) for fixing specific sections

STEP 8: SUGGESTED PLUGINS (stock Logic plugins)
-----------------------------------------------
Guitar tracks:
  - Channel EQ: High-pass around 80Hz, gentle presence boost 3-5kHz
  - Compressor: Ratio 3:1, gentle compression
  - ChromaVerb or Space Designer: Small room, subtle

Vocal track:
  - Channel EQ: High-pass 100Hz, presence at 4kHz
  - Compressor: Ratio 4:1, smooth attack
  - DeEsser: If needed
  - ChromaVerb: Plate or room, 15-20% wet

Master:
  - Keep it simple - this song is intimate
  - Adaptive Limiter at the end, very gentle

================================================================================
"""
    return instructions

def create_project_folder_structure():
    """Create subfolders for organization"""
    folders = [
        'Audio Files',
        'MIDI',
        'Bounces',
        'Notes',
    ]
    
    for folder in folders:
        folder_path = os.path.join(PROJECT_DIR, folder)
        os.makedirs(folder_path, exist_ok=True)
        print(f"Created folder: {folder}")

def create_applescript_helper():
    """Create AppleScript for marker creation"""
    
    applescript = '''-- Logic Pro Marker Setup for Kelly Song
-- Run this after creating your project and setting tempo to 72 BPM

tell application "Logic Pro"
    activate
end tell

tell application "System Events"
    tell process "Logic Pro"
        -- Note: You may need to adjust these based on your Logic Pro version
        -- This script assumes project is open and playhead is at bar 1
        
        -- Create markers using Option+'
        -- You may need to manually position playhead and run marker creation
        
        display dialog "Logic Pro Marker Setup" & return & return & ¬
            "This will help you set up markers." & return & return & ¬
            "Markers to create:" & return & ¬
            "Bar 1: INTRO" & return & ¬
            "Bar 9: VERSE 1" & return & ¬
            "Bar 17: VERSE 2" & return & ¬
            "Bar 25: CHORUS" & return & ¬
            "Bar 33: INSTRUMENTAL" & return & ¬
            "Bar 37: VERSE 3" & return & ¬
            "Bar 45: BRIDGE" & return & ¬
            "Bar 49: FINAL" & return & ¬
            "Bar 54: OUTRO" & return & return & ¬
            "Position playhead at each bar and press Option+' to create marker." ¬
            buttons {"OK"} default button "OK"
    end tell
end tell
'''
    
    script_path = os.path.join(PROJECT_DIR, 'setup_markers.scpt')
    with open(script_path, 'w') as f:
        f.write(applescript)
    print(f"Created: {script_path}")
    
    return script_path

def write_instructions_file():
    """Write the setup instructions to a file"""
    instructions = create_logic_template_instructions()
    
    instructions_path = os.path.join(PROJECT_DIR, 'LOGIC_PRO_SETUP.txt')
    with open(instructions_path, 'w') as f:
        f.write(instructions)
    print(f"Created: {instructions_path}")
    
    return instructions_path

def main():
    print("=" * 50)
    print("Kelly Song Project - Logic Pro Setup")
    print("=" * 50)
    print()
    
    # Create folder structure
    create_project_folder_structure()
    print()
    
    # Write instructions
    write_instructions_file()
    
    # Create AppleScript helper
    create_applescript_helper()
    
    print()
    print("Setup complete!")
    print()
    print("Next steps:")
    print("1. Run: python3 generate_midi.py (if not already done)")
    print("2. Open Logic Pro")
    print("3. Follow instructions in LOGIC_PRO_SETUP.txt")
    print("4. Import kelly_song_reference.mid as a guide track")
    print()

if __name__ == '__main__':
    main()
