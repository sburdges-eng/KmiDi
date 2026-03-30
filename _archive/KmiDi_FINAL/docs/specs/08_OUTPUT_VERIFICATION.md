# 08. Output & Verification Specs

## Overview

Because if it can't leave the app, it didn't happen. These specs ensure users can export their work and verify what they're creating.

## Export UI Spec

### What Can Be Exported

**✓ EXPORTABLE (Standalone Only):**
- MIDI files (generated sequences)
- Audio files (rendered stems/mixes)
- Project files (session state)
- Preset files (reusable settings)

**❌ NOT EXPORTABLE:**
- Raw AI model data
- Internal processing parameters
- Temporary cache files
- Debug logs

**Implementation:**
```cpp
// ExportManager.h - Controlled export capabilities
class ExportManager
{
public:
    enum ExportType { MIDI, Audio, Project, Preset };

    bool canExport (ExportType type) {
        // Standalone app can export everything
        // Plugins cannot export (host handles file I/O)
        return isStandaloneApp();
    }

    void showExportDialog (ExportType type) {
        if (!canExport (type)) return;

        switch (type) {
            case MIDI:
                exportMIDI();
                break;
            case Audio:
                exportAudio();
                break;
            case Project:
                exportProject();
                break;
            case Preset:
                exportPreset();
                break;
        }
    }

private:
    void exportMIDI() {
        // Export generated MIDI sequences
        juce::FileChooser chooser ("Export MIDI File",
                                  juce::File::getSpecialLocation (juce::File::userDesktopDirectory),
                                  "*.mid");
        if (chooser.browseForFileToSave (true)) {
            exportMIDIToFile (chooser.getResult());
        }
    }

    void exportAudio() {
        // Export rendered audio
        juce::FileChooser chooser ("Export Audio File",
                                  juce::File::getSpecialLocation (juce::File::userDesktopDirectory),
                                  "*.wav;*.aiff;*.flac");
        if (chooser.browseForFileToSave (true)) {
            exportAudioToFile (chooser.getResult());
        }
    }
};
```

### Format Selection

**✓ SUPPORTED FORMATS:**
- MIDI: Standard MIDI File (.mid)
- Audio: WAV, AIFF, FLAC (high quality)
- Project: Proprietary format (.daiw)
- Preset: JSON format (.preset)

**❌ UNSUPPORTED:**
- Lossy audio formats (MP3, AAC)
- Proprietary formats without documentation
- Temporary or cache formats

**Implementation:**
```cpp
// ExportFormatSelector.h - Clear format choices
class ExportFormatSelector : public juce::Component
{
public:
    ExportFormatSelector (ExportType type) {
        switch (type) {
            case MIDI:
                addFormat ("Standard MIDI File", ".mid");
                break;
            case Audio:
                addFormat ("WAV (Lossless)", ".wav");
                addFormat ("AIFF (Lossless)", ".aiff");
                addFormat ("FLAC (Compressed)", ".flac");
                // No MP3/AAC - preserve quality
                break;
            case Project:
                addFormat ("DAiW Project", ".daiw");
                break;
            case Preset:
                addFormat ("JSON Preset", ".preset");
                break;
        }
    }

private:
    juce::ComboBox formatSelector;

    void addFormat (const juce::String& name, const juce::String& extension) {
        formatSelector.addItem (name + " (" + extension + ")", formatSelector.getNumItems() + 1);
    }
};
```

### Progress Reporting

**✓ PROGRESS INDICATION:**
- Progress bar for long operations
- Cancel button during export
- Estimated time remaining
- Current operation status

**❌ FORBIDDEN:**
- Blocking UI during export
- No progress indication
- Uncancelable operations

**Implementation:**
```cpp
// ExportProgressDialog.h - Non-blocking progress reporting
class ExportProgressDialog : public juce::Component,
                           private juce::Thread
{
public:
    ExportProgressDialog() : Thread ("Export Thread")
    {
        addAndMakeVisible (progressBar);
        addAndMakeVisible (statusLabel);
        addAndMakeVisible (cancelButton);

        cancelButton.onClick = [this]() {
            if (isThreadRunning()) {
                signalThreadShouldExit();
            }
        };
    }

    void startExport (std::function<void()> exportFunction) {
        exportFunc = exportFunction;
        startThread(); // Non-blocking
    }

private:
    juce::ProgressBar progressBar;
    juce::Label statusLabel;
    juce::TextButton cancelButton;

    std::function<void()> exportFunc;

    void run() override {
        // Run export on background thread
        juce::MessageManagerLock lock; // Safe UI updates

        try {
            exportFunc();
            showCompletionMessage();
        } catch (const std::exception& e) {
            showErrorMessage (e.what());
        }
    }

    void updateProgress (float progress, const juce::String& status) {
        juce::MessageManagerLock lock;
        progressBar.setValue (progress);
        statusLabel.setText (status, juce::dontSendNotification);
    }
};
```

### Error Handling

**✓ ERROR REPORTING:**
- Clear error messages
- Actionable error descriptions
- Recovery suggestions
- No technical jargon

**❌ FORBIDDEN:**
- Generic "export failed" messages
- Technical error codes
- No recovery options
- Modal error dialogs that block

**Implementation:**
```cpp
// ExportErrorHandler.h - User-friendly error handling
class ExportErrorHandler
{
public:
    static void handleExportError (ExportError error, const juce::String& details = "") {
        juce::String title, message;

        switch (error) {
            case FileWriteError:
                title = "Export Failed";
                message = "Unable to save the file. Please check that you have permission to write to the selected location and that there's enough disk space.";
                break;

            case AudioEngineError:
                title = "Audio Export Error";
                message = "There was a problem rendering the audio. Please try exporting a shorter section or check your audio settings.";
                break;

            case MIDIExportError:
                title = "MIDI Export Error";
                message = "Unable to export MIDI data. Please ensure you have MIDI content to export.";
                break;

            case DiskFullError:
                title = "Not Enough Disk Space";
                message = "There's not enough space on your disk to complete the export. Please free up some space and try again.";
                break;
        }

        // Show user-friendly dialog (non-blocking)
        showErrorDialog (title, message + (details.isNotEmpty() ? "\n\nDetails: " + details : ""));
    }

private:
    static void showErrorDialog (const juce::String& title, const juce::String& message) {
        // Non-blocking error notification
        // Could be toast, inline message, or non-modal dialog
    }
};
```

## Preview & Visualization Spec

### Screenshots (Development Only)

**✓ DEVELOPMENT USE:**
- Automated screenshot generation
- UI verification testing
- Documentation assets
- Marked as "development only"

**❌ USER ACCESS:**
- No screenshot functionality in production
- No "save screenshot" menu items
- No screenshot hotkeys

**Implementation:**
```cpp
// ScreenshotManager.h - Development-only screenshots
class ScreenshotManager
{
public:
    static void takeUIScreenshot (const juce::Component& component,
                                 const juce::String& filename)
    {
#ifndef NDEBUG  // Development builds only
        juce::File screenshotFile = getScreenshotDirectory().getChildFile (filename + ".png");

        // Create image of component
        juce::Image screenshot (juce::Image::PixelFormat::RGB, component.getWidth(), component.getHeight(), true);
        juce::Graphics g (screenshot);
        component.paintEntireComponent (g, false);

        // Save to development screenshots directory
        juce::PNGImageFormat pngFormat;
        std::unique_ptr<juce::FileOutputStream> stream (screenshotFile.createOutputStream());
        if (stream != nullptr) {
            pngFormat.writeImageToStream (screenshot, *stream);
        }
#endif
    }

private:
    static juce::File getScreenshotDirectory() {
        return juce::File::getSpecialLocation (juce::File::userApplicationDataDirectory)
                                      .getChildFile ("DAiW")
                                      .getChildFile ("Screenshots");
    }
};
```

### Dev-Only Debug Views

**✓ DEVELOPMENT FEATURES:**
- Performance overlays (FPS, CPU usage)
- Parameter value displays
- AI confidence meters
- Debug logging panels

**❌ PRODUCTION BUILD:**
- All debug views disabled
- No performance impact
- No user-visible debug information

**Implementation:**
```cpp
// DebugVisualization.h - Development-only debug views
class DebugVisualization : public juce::Component
{
public:
    void paint (juce::Graphics& g) override {
#ifndef NDEBUG  // Development builds only
        drawPerformanceOverlay (g);
        drawParameterValues (g);
        drawAIConfidenceMeters (g);
#endif
    }

    void drawPerformanceOverlay (juce::Graphics& g) {
#ifndef NDEBUG
        juce::String perfText = juce::String ("FPS: ") + juce::String (getCurrentFPS()) +
                               "\nCPU: " + juce::String (getCurrentCPUUsage(), 1) + "%";

        g.setColour (juce::Colours::red);
        g.setFont (12.0f);
        g.drawMultiLineText (perfText, 10, 20, getWidth() - 20);
#endif
    }

private:
    // All debug functionality wrapped in #ifndef NDEBUG
};
```

### What Users Never See

**❌ HIDDEN FROM USERS:**
- Raw AI model internals
- Debug logging output
- Performance metrics
- Technical parameter values
- Development screenshots
- Internal processing details

**✓ USER SEES:**
- Musical results
- Clear progress indication
- Helpful error messages
- Intuitive controls
- Professional presentation

**Implementation:**
```cpp
// UserVisibilityFilter.h - What users should and shouldn't see
class UserVisibilityFilter
{
public:
    static bool shouldShowToUser (UIElement element) {
        switch (element) {
            case RawAIModelData:
            case DebugLogs:
            case PerformanceMetrics:
            case TechnicalParameters:
            case DevelopmentScreenshots:
                return false; // Never show to users

            case MusicalResults:
            case ProgressIndicators:
            case HelpfulErrors:
            case IntuitiveControls:
            case ProfessionalUI:
                return true; // Always show to users

            default:
                return false; // Default: hide
        }
    }

    static void filterUIForProduction() {
#ifndef NDEBUG
        // Development build - show everything for testing
        return;
#endif
        // Production build - hide debug elements
        hideDebugOverlays();
        hidePerformanceMeters();
        hideTechnicalParameters();
        disableScreenshotFunctionality();
    }

private:
    enum UIElement {
        RawAIModelData, DebugLogs, PerformanceMetrics, TechnicalParameters,
        DevelopmentScreenshots, MusicalResults, ProgressIndicators,
        HelpfulErrors, IntuitiveControls, ProfessionalUI
    };
};
```

## Audit Checklist

### Export UI Compliance
- [ ] MIDI export works in standalone only
- [ ] Audio export supports WAV/AIFF/FLAC only
- [ ] Project export saves session state
- [ ] Preset export uses JSON format
- [ ] Progress bars shown for long operations
- [ ] Cancel buttons available during export
- [ ] User-friendly error messages
- [ ] Recovery suggestions provided

### Preview & Visualization Compliance
- [ ] Screenshots disabled in production builds
- [ ] Debug views disabled in production builds
- [ ] Performance overlays development-only
- [ ] No technical details shown to users
- [ ] User sees musical results, not internals

### User Visibility Compliance
- [ ] No raw AI model data visible
- [ ] No debug logging output
- [ ] No performance metrics shown
- [ ] No technical parameter displays
- [ ] Professional, intuitive presentation

## Code Examples

### ✅ CORRECT: Export UI Implementation
```cpp
// ExportDialog.h - Complete export UI
class ExportDialog : public juce::Component
{
public:
    ExportDialog() {
        addAndMakeVisible (formatSelector);
        addAndMakeVisible (exportButton);
        addAndMakeVisible (progressBar);
        addAndMakeVisible (statusLabel);

        exportButton.onClick = [this]() {
            startExport();
        };

        // Initially hide progress
        progressBar.setVisible (false);
        statusLabel.setVisible (false);
    }

    void startExport() {
        auto selectedFormat = formatSelector.getSelectedFormat();

        // Show progress UI
        progressBar.setVisible (true);
        statusLabel.setVisible (true);
        exportButton.setEnabled (false);

        // Start async export
        exportManager.startExport (selectedFormat, [this](float progress, const juce::String& status) {
            updateProgress (progress, status);
        }, [this](bool success, const juce::String& error) {
            onExportComplete (success, error);
        });
    }

private:
    ExportFormatSelector formatSelector;
    juce::TextButton exportButton {"Export"};
    juce::ProgressBar progressBar;
    juce::Label statusLabel;

    void updateProgress (float progress, const juce::String& status) {
        progressBar.setValue (progress);
        statusLabel.setText (status, juce::dontSendNotification);
    }

    void onExportComplete (bool success, const juce::String& error) {
        progressBar.setVisible (false);
        exportButton.setEnabled (true);

        if (success) {
            statusLabel.setText ("Export completed successfully!", juce::dontSendNotification);
        } else {
            statusLabel.setText ("Export failed: " + error, juce::dontSendNotification);
            ExportErrorHandler::handleExportError (parseError (error));
        }
    }
};
```

### ✅ CORRECT: Production Build Filtering
```cpp
// ProductionBuildFilter.h - Hide development features in production
class ProductionBuildFilter
{
public:
    static void applyProductionFilters() {
#ifndef NDEBUG
        // Development build - allow all features
        return;
#endif

        // Production build - hide development features
        DebugVisualization::setVisible (false);
        ScreenshotManager::disable();
        PerformanceOverlay::hide();
        TechnicalParameterDisplay::setVisible (false);

        // Ensure user sees professional interface only
        ProfessionalUI::ensureVisible();
        UserFriendlyControls::enable();
        TechnicalDetails::hideAll();
    }

    static void filterMenuItems (juce::PopupMenu& menu) {
#ifndef NDEBUG
        // Development - show all menu items
        return;
#endif

        // Production - remove development menu items
        menu.removeItem (MenuIDs::TakeScreenshot);
        menu.removeItem (MenuIDs::ShowDebugPanel);
        menu.removeItem (MenuIDs::PerformanceMonitor);
        menu.removeItem (MenuIDs::TechnicalParameters);
    }
};
```

### ❌ WRONG: User-Sees-Debug-Info
```cpp
// WRONG - Showing debug info to users
class BadDebugDisplay : public juce::Component
{
    void paint (juce::Graphics& g) override {
        // WRONG: Users see technical debug info
        g.drawText ("AI Model Confidence: 0.87", 10, 20, getWidth(), 20,
                   juce::Justification::left);

        g.drawText ("Processing Latency: 12.3ms", 10, 40, getWidth(), 20,
                   juce::Justification::left);

        g.drawText ("Memory Usage: 156MB", 10, 60, getWidth(), 20,
                   juce::Justification::left);
    }
};
```

## Non-Compliance Fixes

### If Export Functionality Missing:
1. Implement export for MIDI, audio, project, preset formats
2. Add progress reporting for long operations
3. Include cancel functionality
4. Provide clear error messages with recovery options

### If Debug Info Visible in Production:
1. Wrap all debug displays in `#ifndef NDEBUG`
2. Create production build filter to hide debug elements
3. Ensure debug menu items are removed in production
4. Test production builds to verify debug info is hidden

### If Poor Error Handling Found:
1. Replace generic error messages with specific, actionable ones
2. Add recovery suggestions for each error type
3. Implement non-blocking error notifications
4. Test all error conditions with user-friendly messages

### If No Progress Indication:
1. Add progress bars for operations >2 seconds
2. Include cancel buttons for user control
3. Show estimated time remaining
4. Update status messages during operation