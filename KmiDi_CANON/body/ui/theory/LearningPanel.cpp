#include "ui/theory/LearningPanel.h"
#include <juce_graphics/juce_graphics.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_basics/juce_audio_basics.h> // For MidiMessage, MidiBuffer
#include <juce_audio_devices/juce_audio_devices.h> // For MidiOutput
#include <algorithm>
#include <chrono>
#include <thread>

namespace kelly {

LearningPanel::LearningPanel(midikompanion::theory::MusicTheoryBrain* brain)
    : brain_(brain),
      conceptTitle_("", ""),
      explanationDisplay_("Explanation"),
      styleSelector_("Style Selector"),
      styleLabel_("", "Explanation Style"),
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
    loadExplanation(currentConcept_);
    updateExplanationDisplay();
}

void LearningPanel::loadExplanation(const std::string& conceptName)
{
    if (brain_) {
        midikompanion::theory::UserProfile profile{};
        profile.preferredExplanationStyle = currentStyle_;
        auto explanation = brain_->askQuestion("Explain " + conceptName, profile);
        explanationDisplay_.setText(explanation, juce::dontSendNotification);
    }
}

void LearningPanel::updateExplanationDisplay()
{
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

    std::vector<int> notesToPlay;
    if (!activeExampleNotes_.empty()) {
        notesToPlay = activeExampleNotes_;
    } else if (!currentExercise_.notes.empty()) {
        notesToPlay = currentExercise_.notes;
    } else {
        notesToPlay = buildExampleNotes();
    }

    if (notesToPlay.empty()) {
        juce::Logger::writeToLog("LearningPanel: No notes available to play for concept: " + currentConcept_);
        return;
    }

    isPlayingExample_ = true;
    playExampleButton_.setButtonText("Stop Example");

    std::vector<float> onsets = currentExercise_.onsets;
    if (onsets.size() < notesToPlay.size()) {
        onsets.resize(notesToPlay.size(), onsets.empty() ? 0.0f : onsets.back() + 0.5f);
    }

    float lastOnset = 0.0f;
    for (size_t i = 0; i < notesToPlay.size() && isPlayingExample_; ++i) {
        float waitMs = std::max(0.0f, (onsets[i] - lastOnset) * 1000.0f);
        if (waitMs > 0.0f)
            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(waitMs)));

        int note = notesToPlay[i];
        midiOutput_->sendMessageNow(juce::MidiMessage::noteOn(1, note, (juce::uint8)100));
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        midiOutput_->sendMessageNow(juce::MidiMessage::noteOff(1, note));
        lastOnset = onsets[i];
    }

    isPlayingExample_ = false;
    playExampleButton_.setButtonText("Play Example");
}

void LearningPanel::stopExamplePlayback()
{
    if (midiOutput_) {
        for (int channel = 0; channel < 16; ++channel) {
            midiOutput_->sendMessageNow(juce::MidiMessage::allNotesOff(channel + 1));
            midiOutput_->sendMessageNow(juce::MidiMessage::allSoundOff(channel + 1));
        }
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

    midikompanion::theory::UserProfile profile{};
    profile.preferredExplanationStyle = currentStyle_;

    auto session = brain_->generatePracticeSession(profile, 5);
    auto it = std::find_if(session.exercises.begin(), session.exercises.end(),
                           [this](const midikompanion::theory::Exercise& ex) {
                               return ex.conceptName == currentConcept_;
                           });

    if (it != session.exercises.end()) {
        currentExercise_ = *it;
    } else if (!session.exercises.empty()) {
        currentExercise_ = session.exercises.front();
    }

    if (!session.exercises.empty()) {
        displayExercise(currentExercise_);
        hasExercise_ = true;
        nextExerciseButton_.setEnabled(true);
    } else {
        hasExercise_ = false;
        nextExerciseButton_.setEnabled(false);
        explanationDisplay_.setText("No exercises available for " + currentConcept_, juce::dontSendNotification);
    }
}

void LearningPanel::displayExercise(const midikompanion::theory::Exercise& exercise)
{
    juce::String text;
    text << "\n-- Exercise: " << exercise.conceptName << " --\n\n";
    text << "Description: " << exercise.instruction << "\n\n";
    text << "Focus Area: " << exercise.focusArea << "\n";

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
    if (midiOutput_) {
        return true;
    }

    auto devices = juce::MidiOutput::getAvailableDevices();
    if (devices.isEmpty()) {
        juce::Logger::writeToLog("LearningPanel: No MIDI output devices found.");
        return false;
    }

    midiOutput_ = juce::MidiOutput::openDevice(devices[0].identifier);
    if (!midiOutput_) {
        juce::Logger::writeToLog("LearningPanel: Failed to open default MIDI output device.");
        return false;
    }
    return true;
}

std::vector<int> LearningPanel::buildExampleNotes() const
{
    return {60, 62, 64, 65, 67, 69, 71, 72};
}

} // namespace kelly
