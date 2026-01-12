# KmiDi Qt GUI User Manual

**Version**: 1.0.0

## Overview

The KmiDi Qt GUI provides a professional desktop interface for emotion-driven music generation. It combines an intuitive interface with powerful music generation capabilities.

## Installation

### macOS

```bash
# Install dependencies
pip install PySide6>=6.5.0

# Run the application
python kmidi_gui/main.py
```

### Linux

```bash
# Install dependencies
pip install PySide6>=6.5.0

# Run the application
python kmidi_gui/main.py
```

### Windows

```powershell
# Install dependencies
pip install PySide6>=6.5.0

# Run the application
python kmidi_gui\main.py
```

## Getting Started

### Launching the Application

```bash
# From project root
python kmidi_gui/main.py

# Or as module
python -m kmidi_gui.main
```

### First Run

1. **Check API Status**: The status bar shows API connection status
2. **Enter Emotion**: Type your emotional intent in the text area
3. **Generate**: Click the "Generate" button
4. **Preview**: Listen to the generated music
5. **Export**: Save MIDI or audio files

## Interface Overview

### Main Window

The main window consists of:

1. **Menu Bar**: File, Edit, View, Tools, Help menus
2. **Toolbar**: Quick access buttons (Generate, Preview, Export)
3. **Central Area**: 
   - Emotion input text area
   - Results display area
   - Parameter controls
4. **Status Bar**: API status, generation progress, messages

### Menu Bar

#### File Menu

- **New Project** - Start a new music generation project
- **Open Project** - Load a saved project
- **Save Project** - Save current project
- **Export MIDI** - Export generated MIDI file
- **Exit** - Quit application

#### Edit Menu

- **Undo** - Undo last action
- **Redo** - Redo last action
- **Preferences** - Open preferences dialog

#### View Menu

- **Show AI Assistant** - Toggle AI assistant dock
- **Show Logs** - Toggle logs dock

#### Tools Menu

- **AI Analysis** - Run AI analysis on current project
- **Batch Process** - Process multiple files

#### Help Menu

- **About** - Application information
- **Documentation** - Open documentation

### Toolbar

- **Generate Button** - Generate music from current emotion input
- **Preview Button** - Preview generated music
- **Export Button** - Export current generation

### Central Widget

#### Emotion Input

Enter your emotional intent as free-form text:

- Examples:
  - "grief hidden as love"
  - "anxious and worried"
  - "calm and peaceful"
  - "joyful celebration"

#### Technical Parameters

Adjust technical parameters (optional):

- **Key**: Musical key (C, D, E, F, G, A, B)
- **BPM**: Tempo in beats per minute (60-180)
- **Genre**: Music genre (pop, rock, jazz, electronic, acoustic)
- **Mode**: Major or minor

#### Results Display

After generation, view:

- **Chord Progression**: Generated chords
- **Key**: Detected key
- **Tempo**: Generated tempo
- **MIDI Path**: Location of generated MIDI file

### Status Bar

Shows:

- **API Status**: Connection status (Online/Offline/Checking...)
- **Generation Status**: Current operation status
- **Messages**: Information and error messages

## Workflows

### Basic Music Generation

1. **Enter Emotion**: Type emotional intent in text area
2. **Adjust Parameters** (optional): Set key, BPM, genre
3. **Generate**: Click "Generate" button
4. **Wait**: Status bar shows progress
5. **Preview**: Click "Preview" to listen
6. **Export**: Click "Export" to save

### Advanced Workflow

1. **Create Project**: File → New Project
2. **Enter Multiple Emotions**: Try different emotional intents
3. **Compare Results**: Generate multiple versions
4. **Refine Parameters**: Adjust technical parameters
5. **Save Project**: File → Save Project
6. **Export Final**: Export MIDI or audio

### Batch Processing

1. **Prepare Files**: Organize input files
2. **Tools → Batch Process**: Open batch processing dialog
3. **Select Files**: Choose files to process
4. **Set Parameters**: Configure processing options
5. **Process**: Start batch processing
6. **Review Results**: Check generated files

## Features

### Emotion-to-Music Generation

- **Natural Language Input**: Describe emotions in plain language
- **Emotional Presets**: Quick selection from predefined emotions
- **Custom Emotions**: Enter any emotional description

### Music Analysis

- **Chord Analysis**: Analyze chord progressions
- **Key Detection**: Automatic key detection
- **Tempo Analysis**: BPM detection and analysis

### Preview and Export

- **Audio Preview**: Listen before exporting
- **MIDI Export**: Export as MIDI file
- **Audio Export**: Export as audio file (if configured)

### AI Assistant (Coming Soon)

- **Interactive Help**: Get suggestions and guidance
- **Music Theory**: Learn about music theory
- **Composition Tips**: Get composition suggestions

## Keyboard Shortcuts

- **Ctrl+N** (Cmd+N on macOS): New Project
- **Ctrl+O** (Cmd+O on macOS): Open Project
- **Ctrl+S** (Cmd+S on macOS): Save Project
- **Ctrl+G** (Cmd+G on macOS): Generate Music
- **Ctrl+P** (Cmd+P on macOS): Preview
- **Ctrl+E** (Cmd+E on macOS): Export
- **F1**: Help
- **Esc**: Close dialogs

## Configuration

### Preferences

Access via **Edit → Preferences**:

- **API Settings**: Configure API endpoint
- **Audio Settings**: Configure audio output
- **Theme**: Select application theme
- **Advanced**: Advanced configuration options

### Environment Variables

See [Environment Configuration](ENVIRONMENT_CONFIGURATION.md) for all options.

Key variables:
- `KELLY_AUDIO_DATA_ROOT` - Audio data directory
- `LOG_LEVEL` - Logging level

## Troubleshooting

### Application Won't Start

1. **Check Python Version**: Requires Python 3.9+
   ```bash
   python --version
   ```

2. **Check Dependencies**:
   ```bash
   pip install PySide6>=6.5.0
   ```

3. **Check Logs**: Look for error messages in console

### API Connection Failed

1. **Check API Status**: Status bar shows connection status
2. **Verify API Running**: 
   ```bash
   curl http://localhost:8000/health
   ```
3. **Check Configuration**: Verify API endpoint in preferences

### Generation Fails

1. **Check Input**: Ensure emotion text is entered
2. **Check API**: Verify API is running and accessible
3. **Check Logs**: Review error messages in status bar
4. **Try Simple Input**: Test with simple emotion like "calm"

### Preview Not Working

1. **Check Audio System**: Verify audio output is configured
2. **Check File**: Ensure MIDI file was generated
3. **Check Permissions**: Verify file read permissions

## Tips and Best Practices

1. **Start Simple**: Begin with simple emotions like "calm" or "sad"
2. **Experiment**: Try different emotional descriptions
3. **Use Parameters**: Adjust technical parameters for fine-tuning
4. **Save Projects**: Save your work frequently
5. **Review Results**: Listen to generated music before exporting

## Advanced Features

### Custom Themes

The application supports custom themes. Edit theme files in `kmidi_gui/themes/`.

### Plugin System (Coming Soon)

Extend functionality with plugins:
- Custom generators
- Additional analysis tools
- Integration with DAWs

## Support

For issues or questions:
- Check logs: View application logs for errors
- Review documentation: See other documentation files
- Report issues: Create an issue on GitHub

## See Also

- [CLI Usage Guide](CLI_USAGE_GUIDE.md)
- [API Documentation](API_DOCUMENTATION.md)
- [Quick Start Guide](QUICKSTART_GUIDE.md)
