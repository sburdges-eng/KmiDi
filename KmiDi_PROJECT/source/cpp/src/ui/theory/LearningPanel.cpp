#include "ui/theory/LearningPanel.h"
#include <juce_graphics/juce_graphics.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_basics/juce_audio_basics.h> // For MidiMessage, MidiBuffer
#include <juce_audio_devices/juce_audio_devices.h> // For MidiOutput
#include <algorithm>
#include <chrono>
#include <thread>

namespace kelly {

// Forward declaration for MusicTheoryBrain::MidiMessage if it's different from juce::MidiMessage
// In Types.h, midikompanion::theory::MidiMessage is defined.
// We need to convert it to juce::MidiMessage for playback.

LearningPanel::LearningPanel(midikompanion::theory::MusicTheoryBrain* brain)
    : brain_(brain),
      conceptTitle_("", ""),
      explanationDisplay_("Explanation"), // Corrected constructor
      styleSelector_("Style Selector"),
      styleLabel_{"", "Explanation Style"},
      playExampleButton_("Play Example"),
      nextExerciseButton_("Next Exercise")
{
    setupComponents();
}

void LearningPanel::setupComponents()
{
    addAndMakeVisible(conceptTitle_);
    conceptTitle_.setFont(juce::Font(24.0f, juce::Font::bold));
    conceptTitle_.setJustificationType(juce::Justification::centred);

    addAndMakeVisible(explanationDisplay_);
    explanationDisplay_.setMultiLine(true, true);
    explanationDisplay_.setReturnKeyStartsNewLine(false);
    explanationDisplay_.setReadOnly(true);
    explanationDisplay_.setScrollbarsShown(true);
    explanationDisplay_.setCaretVisible(false);

    addAndMakeVisible(styleLabel_);
    addAndMakeVisible(styleSelector_);
    styleSelector_.addItem("Intuitive", (int)midikompanion::theory::ExplanationType::Intuitive + 1);
    styleSelector_.addItem("Mathematical", (int)midikompanion::theory::ExplanationType::Mathematical + 1);
    styleSelector_.addItem("Historical", (int)midikompanion::theory::ExplanationType::Historical + 1);
    styleSelector_.setSelectedId((int)currentStyle_ + 1);
    styleSelector_.onChange = [this] {
        setExplanationStyle((midikompanion::theory::ExplanationType)(styleSelector_.getSelectedId() - 1));
    };

    playExampleButton_.onClick = [this] { playCurrentConceptExample(); };
    addAndMakeVisible(playExampleButton_);

    nextExerciseButton_.onClick = [this] { loadNextExercise(); };
    addAndMakeVisible(nextExerciseButton_);

    // Attempt to initialize MIDI output
    ensureMidiOutputReady();
}

void LearningPanel::paint(juce::Graphics& g)
{
    g.fillAll(getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId));
}

void LearningPanel::resized()
{
    juce::Rectangle<int> bounds = getLocalBounds().reduced(10);
    int y = bounds.getY();
    int height = 30;

    conceptTitle_.setBounds(bounds.getX(), y, bounds.getWidth(), height);
    y += height + 10;

    styleLabel_.setBounds(bounds.getX(), y, 100, height);
    styleSelector_.setBounds(bounds.getX() + 105, y, 150, height);
    y += height + 10;

    playExampleButton_.setBounds(bounds.getX(), y, 120, height);
    nextExerciseButton_.setBounds(playExampleButton_.getRight() + 10, y, 120, height);
    y += height + 10;

    explanationDisplay_.setBounds(bounds.getX(), y, bounds.getWidth(), bounds.getHeight() - y);
}

void LearningPanel::displayConcept(const std::string& conceptName)
{
    currentConcept_ = conceptName;
    conceptTitle_.setText(conceptName, juce::dontSendNotification);
    loadExplanation(conceptName);
    // loadNextExercise(); // Load first exercise when concept displayed - defer this until requested
    updateExplanationDisplay();
}

void LearningPanel::displayExplanation(const std::string& text, midikompanion::theory::ExplanationType style)
{
    currentStyle_ = style;
    styleSelector_.setSelectedId((int)currentStyle_ + 1);
    explanationDisplay_.setText(text, juce::dontSendNotification);
}

void LearningPanel::setMusicTheoryBrain(midikompanion::theory::MusicTheoryBrain* brain)
{
    brain_ = brain;
}

void LearningPanel::setExplanationStyle(midikompanion::theory::ExplanationType style)
{
    currentStyle_ = style;
    loadExplanation(currentConcept_); // Reload explanation with new style
    updateExplanationDisplay();
}

void LearningPanel::loadExplanation(const std::string& conceptName)
{
    if (brain_) {
        // MusicTheoryBrain::getConceptExplanation requires ExplanationDepth
        auto explanation = brain_->getConceptExplanation(conceptName, currentStyle_, midikompanion::theory::ExplanationDepth::Intermediate);
        if (explanation.has_value()) {
            explanationDisplay_.setText(explanation.value(), juce::dontSendNotification);
        } else {
            explanationDisplay_.setText("Explanation not found for " + conceptName, juce::dontSendNotification);
        }
    }
}

void LearningPanel::updateExplanationDisplay()
{
    // This method is called after style changes or concept loads to refresh the text
    loadExplanation(currentConcept_);
}

void LearningPanel::playCurrentConceptExample()
{
    if (isPlayingExample_) {
        stopExamplePlayback();
        return;
    }

    if (!brain_ || currentConcept_.empty()) {
        juce::Logger::writeToLog("LearningPanel: Cannot play example, brain or concept not set.");
        return;
    }

    if (!ensureMidiOutputReady()) {
        juce::Logger::writeToLog("LearningPanel: MIDI output not ready.");
        return;
    }

    // Create a dummy UserProfile for getConceptExample
    midikompanion::theory::UserProfile dummyProfile;

    auto midiExample = brain_->getConceptExample(currentConcept_, dummyProfile);
    if (midiExample) {
        // Convert the vector of MIDI messages to a juce::MidiBuffer
        juce::MidiBuffer buffer;
        for (const auto& midiMsg : midiExample.value()) {
            // Assume midiMsg is midikompanion::theory::MidiMessage, defined in Types.h
            juce::MidiMessage juceNoteOn = juce::MidiMessage::noteOn(1, midiMsg.pitch, (uint8)midiMsg.velocity);
            buffer.addEvent(juceNoteOn, static_cast<int>(midiMsg.startTime * 480)); // 480 ticks per beat
            juce::MidiMessage juceNoteOff = juce::MidiMessage::noteOff(1, midiMsg.pitch, (uint8)0);
            buffer.addEvent(juceNoteOff, static_cast<int>((midiMsg.startTime + midiMsg.duration) * 480));
        }

        // Play the MIDI buffer (simplified, in a real app this would be a MIDI player thread)
        for (const auto& msg : buffer) {
            midiOutput_->sendMessageNow(msg.getMessage());
            std::this_thread::sleep_for(std::chrono::milliseconds(50)); // Small delay for playback
        }

        isPlayingExample_ = true;
        playExampleButton_.setButtonText("Stop Example");
    } else {
        juce::Logger::writeToLog("LearningPanel: No MIDI example available for concept: " + currentConcept_);
    }
}

void LearningPanel::stopExamplePlayback()
{
    if (midiOutput_) {
        // Send all notes off (simplified)
        for (int channel = 0; channel < 16; ++channel) {
            midiOutput_->sendMessageNow(juce::MidiMessage::allNotesOff(channel + 1));
            midiOutput_->sendMessageNow(juce::MidiMessage::allSoundOff(channel + 1));
        }
        // midiOutput_.reset(); // Do not reset, keep device open for subsequent plays
    }
    isPlayingExample_ = false;
    playExampleButton_.setButtonText("Play Example");
}

void LearningPanel::loadNextExercise()
{
    if (!brain_ || currentConcept_.empty()) {
        hasExercise_ = false;
        nextExerciseButton_.setEnabled(false);
        explanationDisplay_.setText("No exercises available.", juce::dontSendNotification);
        return;
    }

    // Create a dummy UserProfile for getNextExercise
    midikompanion::theory::UserProfile dummyProfile;

    auto exerciseOpt = brain_->getNextExercise(currentConcept_, dummyProfile);
    if (exerciseOpt.has_value()) {
        currentExercise_ = exerciseOpt.value();
        displayExercise(currentExercise_);
        hasExercise_ = true;
        nextExerciseButton_.setEnabled(true);
    } else {
        hasExercise_ = false;
        nextExerciseButton_.setEnabled(false);
        explanationDisplay_.setText("No more exercises available for " + currentConcept_, juce::dontSendNotification);
    }
}

void LearningPanel::displayExercise(const midikompanion::theory::Exercise& exercise)
{
    juce::String text;
    // Corrected member access for Exercise (based on Types.h)
    text << "\n-- Exercise: " << exercise.conceptName << " --\n\n";
    text << "Description: " << exercise.instruction << "\n\n"; // Renamed from description to instruction
    text << "Focus Area: " << exercise.focusArea << "\n";

    // Handle MIDI example if available in exercise (e.g., from notes/onsets)
    if (!exercise.notes.empty()) {
        text << "\n(Interactive MIDI example available for this exercise)";
        activeExampleNotes_ = exercise.notes;
    } else {
        activeExampleNotes_.clear();
    }

    explanationDisplay_.setText(text, juce::dontSendNotification);
}

bool LearningPanel::ensureMidiOutputReady()
{
    if (midiOutput_ && midiOutput_->isOpen()) {
        return true;
    }

    auto devices = juce::MidiOutput::getAvailableDevices();
    if (devices.isEmpty()) {
        juce::Logger::writeToLog("LearningPanel: No MIDI output devices found.");
        return false;
    }
    // Attempt to open the default device (first one)
    midiOutput_ = juce::MidiOutput::openDevice(devices[0].identifier);
    if (!midiOutput_) {
        juce::Logger::writeToLog("LearningPanel: Failed to open default MIDI output device.");
        return false;
    }
    return true;
}

std::vector<int> LearningPanel::buildExampleNotes() const
{
    // This is a placeholder for building notes for a virtual keyboard or display.
    // In a real scenario, this would involve parsing the current concept's MIDI example.
    return {60, 62, 64, 65, 67, 69, 71, 72}; // C Major Scale as a dummy example
}

} // namespace kelly
