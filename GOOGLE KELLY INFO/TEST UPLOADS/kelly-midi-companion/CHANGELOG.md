# Changelog

All notable changes to Kelly MIDI Companion will be documented in this file.

## [2.0.0] - 2025-12-09

### Added
- **Full Cassette Visual Design**: Complete v2.0 visual implementation
  - Animated tape reels with rotation
  - Realistic cassette body with gradient and texture
  - Tape window showing content
  - Custom label area with text
  - Toggleable cassette view wrapper
- **Emotion Wheel Selector**: Visual emotion selection component
  - Circular wheel organized by valence (angle) and arousal (radius)
  - Interactive emotion points with hover and selection
  - Real-time emotion name display
  - Automatic slider updates when emotion is selected
  - Toggle button to show/hide wheel
- **Voice Synthesis Integration**: Framework for vocal melody generation
  - `VoiceSynthesizer` class with vocal note generation
  - Emotion-based melody contour generation
  - Lyric generation based on emotion and wound
  - Vocal characteristics mapping (brightness, breathiness, vibrato)
  - Audio synthesis framework (ready for vocoder integration)
- **Biometric Input Support**: Real-time biometric data integration
  - `BiometricInput` class for processing biometric data
  - Heart rate → arousal mapping
  - Skin conductance → intensity mapping
  - Movement → arousal mapping
  - Temperature → valence estimation
  - Data smoothing and averaging
  - Callback system for real-time updates
- **Enhanced UI Integration**: v2.0 components integrated into PluginEditor
  - Emotion wheel toggle button
  - Cassette view toggle button
  - Seamless integration with existing controls

### Changed
- **CassetteView**: Enhanced with full v2.0 visual design
  - Animated reels with rotation
  - Better visual depth and realism
  - Tape position tracking
- **PluginEditor**: Added v2.0 feature toggles and layout
  - Emotion wheel integration
  - Cassette view wrapper option
  - Improved layout for new components

### Technical
- New UI components: `EmotionWheel`, enhanced `CassetteView`
- New engine components: `VoiceSynthesizer`, `BiometricInput`
- Framework ready for hardware integration (biometric sensors)
- Framework ready for vocoder integration (voice synthesis)

---

## [1.5.0] - 2025-12-09

### Added
- **WoundProcessor**: Extracted wound processing logic from IntentPipeline into dedicated component
  - Keyword-based emotion detection
  - Fallback to valence/arousal estimation
  - Better separation of concerns
- **RuleBreakEngine**: Extracted rule-breaking logic from IntentPipeline into dedicated component
  - Emotion-based rule break generation
  - Journey-specific rule breaks for emotional transitions
  - Severity calculation based on emotion intensity
- **UI Components**: Implemented visual components for cassette aesthetic
  - **CassetteView**: Main visual container with cassette tape frame, holes, and label area
  - **SidePanel**: Panel component for Side A/B metaphor with custom styling
  - **GenerateButton**: Custom styled button with rounded corners and hover effects
  - **KellyLookAndFeel**: Custom look and feel with cassette color scheme
- **Test Suite**: Basic test infrastructure with Catch2
  - Tests for WoundProcessor
  - Tests for RuleBreakEngine
  - Test structure for future expansion

### Changed
- **IntentPipeline**: Refactored to use WoundProcessor and RuleBreakEngine
  - Cleaner architecture with better separation of concerns
  - Easier to test and maintain individual components
- **Code Organization**: Improved modularity and component separation

### Technical
- Better code architecture with dedicated components
- Test infrastructure in place for quality assurance
- UI components ready for integration (currently optional)

---

## [1.1.0] - 2025-12-09

### Added
- **Full GrooveEngine Implementation**: Complete port from Python groove engine
  - Timing drift and micro-timing adjustments with human latency bias
  - Velocity humanization with vulnerability-based shaping
  - Note dropouts based on complexity level
  - Ghost notes for natural feel
  - Emotion-based timing adjustments (sad emotions drag, angry rush ahead)
  - Support for multiple groove types: Straight, Swing, Syncopated, Shuffle, Halftime
- **24 Additional Emotion Presets**: Expanded from 36 to 60 presets
  - Complex emotions: Bittersweet, Melancholy, Euphoria, Catharsis, Yearning, Triumph, Despair, Ecstasy
  - Emotional states: Resignation, Determination, Vulnerability, Empowerment, Isolation, Connection
  - Release, Tension, Relief, Anticipation, Fulfillment, Emptiness, Wholeness, Fragility, Resilience, Transcendence
- **Preset Import/Export**: Save and load custom emotion configurations
  - Export current settings to JSON files
  - Import presets from JSON files
  - Preset directory: `~/Library/Application Support/Kelly MIDI Companion/Presets/`
- **Real-time Parameter Automation**: Full support for DAW automation
  - All parameters (Valence, Arousal, Intensity, Complexity, Feel, Dynamics, Bars) can be automated
  - JUCE's AudioProcessorValueTreeState handles automation automatically
- **Enhanced DAW Transport Sync**: Improved timing accuracy
  - Accurate BPM tracking from DAW playhead
  - Better beat position synchronization
  - More precise MIDI note scheduling

### Changed
- GrooveEngine is now fully functional (was previously a stub)
- Increased number of presets from 37 to 61 (60 emotions + 1 default)
- Improved timing accuracy in MIDI playback

### Technical
- GrooveEngine uses Mersenne Twister RNG for humanization
- Preset files use JSON format for portability
- Parameter automation leverages JUCE's built-in automation system

---

## [1.0.0] - 2025-12-09

### Added
- 216-node emotion thesaurus integration
- Three-phase intent pipeline (Wound → Emotion → Rule-Breaks)
- Full MIDI generation system:
  - Chord progression generation
  - Melody generation based on emotion and complexity
  - Bass line generation
  - Groove and humanization
- Emotion-based presets (36 presets)
- Category and style selection system (similar to Logic Pro Session Player)
- Fine-tuning controls:
  - Complexity slider
  - Feel slider (pull/push timing)
  - Dynamics slider
  - Bars selector
- Immediate MIDI playback (no need to wait for DAW transport)
- MIDI file export with automatic Finder reveal on macOS
- Temporary MIDI cache folder in `~/Music/Kelly MIDI Companion/`
- Resizable plugin editor window
- macOS Gatekeeper bypass (code signing and quarantine removal)

### Changed
- Removed Side A/B metaphor, replaced with single unified controls
- Updated UI to use 3 primary sliders (Valence, Arousal, Intensity)
- Sliders update automatically based on selected emotion preset
- Improved plugin description for accuracy

### Fixed
- macOS Gatekeeper "damaged" plugin error
- MIDI notes now visible in Logic Pro MIDI regions
- All markdownlint issues resolved
- Bounds checking for empty chord pitches
- Division by zero safety checks
- Thread-safe MIDI scheduling

### Technical
- Built with JUCE 8.0.4
- C++20 standard
- CMake build system
- Supports VST3, AU, and Standalone formats

---

## Future Versions

### v2.1 (Planned)
- [ ] Full vocoder integration for voice synthesis
- [ ] Hardware biometric sensor integration
- [ ] Python-C++ bridge for advanced features
- [ ] Advanced emotion wheel with 3D visualization
