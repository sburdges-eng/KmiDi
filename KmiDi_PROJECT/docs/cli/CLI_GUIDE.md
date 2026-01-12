# DAiW-Music-Brain Command Line Interface (CLI) Guide

This guide provides a reference for the `daiw` command-line interface, part of the DAiW-Music-Brain toolkit. This CLI allows for emotion-driven music generation, audio analysis, and interactive teaching directly from your terminal.

## Table of Contents
1.  [Installation](#1-installation)
2.  [Commands](#2-commands)
    *   [2.1 `daiw extract`](#21-daiw-extract)
    *   [2.2 `daiw apply`](#22-daiw-apply)
    *   [2.3 `daiw analyze`](#23-daiw-analyze)
    *   [2.4 `daiw diagnose`](#24-daiw-diagnose)
    *   [2.5 `daiw intent`](#25-daiw-intent)
    *   [2.6 `daiw teach`](#26-daiw-teach)
    *   [2.7 `daiw generate`](#27-daiw-generate)
3.  [Examples](#3-examples)
4.  [Troubleshooting](#4-troubleshooting)

---

## 1. Installation

To use the `daiw` CLI, ensure the KmiDi project's Python environment is set up. From the project root, run:

```bash
pip install -e ".[all]" # Installs all dependencies, including CLI tools
```

This makes the `daiw` command available in your terminal.

---

## 2. Commands

### 2.1 `daiw extract <midi_file> [--output <json_file>]`

Extracts groove information from a MIDI file.

*   **`<midi_file>`**: Path to the input MIDI file.
*   **`--output <json_file>`**: (Optional) Path to save the extracted groove data as a JSON file. If not provided, a default name will be used.

### 2.2 `daiw apply --genre <genre_name> <midi_file> [--output <midi_file>]`

Applies a specified genre's groove template to a MIDI file.

*   **`--genre <genre_name>`**: The name of the genre template to apply (e.g., `funk`, `rock`, `jazz`).
*   **`<midi_file>`**: Path to the input MIDI file.
*   **`--output <midi_file>`**: (Optional) Path to save the modified MIDI file. If not provided, a default name will be used.

### 2.3 `daiw analyze <midi_file> --chords`

Analyzes a MIDI file to detect chord progressions and other musical structures.

*   **`<midi_file>`**: Path to the input MIDI file.
*   **`--chords`**: Flag to analyze and display the detected chord progression.

### 2.4 `daiw diagnose <progression>`

Diagnoses potential harmonic or compositional issues in a given chord progression.

*   **`<progression>`**: The chord progression to diagnose, in a comma-separated or hyphenated string (e.g., `C-G-Am-F`).

### 2.5 `daiw intent <subcommand> [...]`

Manages emotional intents for music generation. Supports subcommands:

*   **`daiw intent new --title "<title>"`**
    *   Creates a new emotional intent template.
    *   **`--title <title>`**: Title for the new intent (e.g., "My Song").
*   **`daiw intent suggest <topic>`**
    *   Suggests rule-breaking options for a given emotional topic.
    *   **`<topic>`**: The emotional topic for suggestions (e.g., `grief`, `anxiety`).

### 2.6 `daiw teach <topic>`

Starts an interactive teaching mode for various music theory or composition topics.

*   **`<topic>`**: The topic for the interactive teaching session (e.g., `rulebreaking`, `voice_leading`).

### 2.7 `daiw generate <emotion_text>`

Generates music directly from a natural language emotional intent.

*   **`<emotion_text>`**: A text description of the emotional intent (e.g., "I feel peaceful and calm, like a quiet morning").

---

## 3. Examples

```bash
# Extract groove from a MIDI file
daiw extract drums.mid --output my_groove.json

# Apply a funk genre template
daiw apply --genre funk track.mid --output funk_track.mid

# Analyze chords in a song
daiw analyze song.mid --chords

# Diagnose a chord progression
daiw diagnose "C-G-Am-F"

# Create a new intent template
daiw intent new --title "Melancholy Ballad"

# Get rule-break suggestions for grief
daiw intent suggest grief

# Generate music from a specific emotion
daiw generate "I feel serene and hopeful"
```

---

## 4. Troubleshooting

*   **`daiw: command not found`**: Ensure you have run `pip install -e ".[all]"` from the project root and your Python environment's scripts directory is in your system's PATH.
*   **`Error: MIDI file not found`**: Verify the path to your MIDI file is correct.
*   **`Unknown genre` / `Unknown topic`**: Check the available genres/topics by consulting project documentation or source code for valid options.
*   **API Errors from `daiw generate`**: Ensure the FastAPI Music Generation API is running (typically on `http://127.0.0.1:8000`). If it's not, start it manually: `cd KMiDi_PROJECT/api && uvicorn main:app --reload --host 127.0.0.1 --port 8000`.
