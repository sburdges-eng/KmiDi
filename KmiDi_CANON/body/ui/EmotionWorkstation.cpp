#include "EmotionWorkstation.h"
#include "../voice/LyricTypes.h"
#include "plugin/PluginProcessor.h"

namespace kelly {

EmotionWorkstation::EmotionWorkstation(
    juce::AudioProcessorValueTreeState &apvts)
    : apvts_(apvts) {
  setupComponents();
}

void EmotionWorkstation::setupComponents() {
  // Apply custom look and feel
  setLookAndFeel(&lookAndFeel_);

  // ========================================================================
  // WOUND INPUT
  // ========================================================================
  woundLabel_.setFont(juce::Font(16.0f, juce::Font::bold));
  woundLabel_.setJustificationType(juce::Justification::centredLeft);
  woundLabel_.setAccessible(true);
  addAndMakeVisible(woundLabel_);

  woundInput_.setMultiLine(true);
  woundInput_.setReturnKeyStartsNewLine(true);
  woundInput_.setPopupMenuEnabled(true);
  woundInput_.setTextToShowWhenEmpty(
      "Describe what you're feeling...",
      lookAndFeel_.textSecondary.withAlpha(0.6f));
  woundInput_.setFont(juce::Font(14.0f));
  woundInput_.setTitle("Wound input field");
  woundInput_.setDescription(
      "Describe your emotional state or wound for processing");
  woundInput_.onTextChange = [this]() {
    if (onWoundTextChanged) {
      onWoundTextChanged(woundInput_.getText());
    }
  };
  addAndMakeVisible(woundInput_);

  emotionLabel_.setFont(juce::Font(13.0f, juce::Font::bold));
  emotionLabel_.setJustificationType(juce::Justification::centredLeft);
  emotionLabel_.setAccessible(true);
  addAndMakeVisible(emotionLabel_);

  emotionInput_.setMultiLine(false);
  emotionInput_.setReturnKeyStartsNewLine(false);
  emotionInput_.setPopupMenuEnabled(true);
  emotionInput_.setTextToShowWhenEmpty(
      "e.g. anxious, hopeful, calm",
      lookAndFeel_.textSecondary.withAlpha(0.6f));
  emotionInput_.setFont(juce::Font(13.0f));
  emotionInput_.setTitle("Emotion entry field");
  emotionInput_.setDescription(
      "Type an emotion name to select it directly (matches thesaurus)");
  emotionInput_.onReturnKey = [this]() {
    if (onEmotionTextCommitted) {
      onEmotionTextCommitted(emotionInput_.getText());
    }
  };
  emotionInput_.onFocusLost = [this]() {
    if (onEmotionTextCommitted) {
      onEmotionTextCommitted(emotionInput_.getText());
    }
  };
  addAndMakeVisible(emotionInput_);

  previewLabel_.setFont(juce::Font(13.0f, juce::Font::bold));
  previewLabel_.setJustificationType(juce::Justification::centredLeft);
  previewLabel_.setAccessible(true);
  addAndMakeVisible(previewLabel_);

  previewText_.setMultiLine(true);
  previewText_.setReadOnly(true);
  previewText_.setScrollbarsShown(true);
  previewText_.setCaretVisible(false);
  previewText_.setFont(juce::Font(12.5f));
  previewText_.setTitle("Preview summary");
  previewText_.setDescription(
      "Preview of how current settings may influence the output");
  addAndMakeVisible(previewText_);

  // ========================================================================
  // EMOTION MAPPING (LEFT PANEL)
  // ========================================================================
  emotionWheel_.onEmotionSelected([this](const EmotionNode &emotion) {
    handleEmotionWheelSelection(emotion);
  });
  emotionWheel_.setTitle("Emotion Wheel");
  emotionWheel_.setDescription(
      "216-node emotion selector - click to select an emotion");
  addAndMakeVisible(emotionWheel_);

  emotionRadar_.setTitle("Emotion Radar");
  emotionRadar_.setDescription("Valence/Arousal/Intensity visualization");
  addAndMakeVisible(emotionRadar_);

  // ========================================================================
  // MUSICAL PARAMETERS (RIGHT PANEL) - ALL 9 APVTS PARAMETERS
  // ========================================================================
  setupSlider(valenceSlider_, valenceLabel_, "Valence");
  valenceSlider_.setRange(-1.0, 1.0, 0.01);
  valenceSlider_.setTextValueSuffix(" (negative ← → positive)");
  valenceSlider_.setDescription(
      "Emotional valence: negative (sad/angry) to positive (happy/joyful)");

  setupSlider(arousalSlider_, arousalLabel_, "Arousal");
  arousalSlider_.setRange(0.0, 1.0, 0.01);
  arousalSlider_.setTextValueSuffix(" (calm → excited)");
  arousalSlider_.setDescription(
      "Emotional arousal: calm/low energy to excited/high energy");

  setupSlider(intensitySlider_, intensityLabel_, "Intensity");
  intensitySlider_.setRange(0.0, 1.0, 0.01);
  intensitySlider_.setTextValueSuffix(" (subtle → intense)");
  intensitySlider_.setDescription(
      "Emotional intensity: subtle feeling to overwhelming emotion");

  setupSlider(complexitySlider_, complexityLabel_, "Complexity");
  complexitySlider_.setRange(0.0, 1.0, 0.01);
  complexitySlider_.setTextValueSuffix(" (simple → complex)");
  complexitySlider_.setDescription(
      "Musical complexity: simple patterns to complex arrangements");

  setupSlider(humanizeSlider_, humanizeLabel_, "Humanize");
  humanizeSlider_.setRange(0.0, 1.0, 0.01);
  humanizeSlider_.setTextValueSuffix(" (rigid → human)");
  humanizeSlider_.setDescription(
      "Humanization: perfect timing to natural human feel");

  setupSlider(feelSlider_, feelLabel_, "Feel");
  feelSlider_.setRange(-1.0, 1.0, 0.01);
  feelSlider_.setTextValueSuffix(" (laid back ← → pushed)");
  feelSlider_.setDescription(
      "Timing feel: laid back/behind the beat to pushed/ahead of the beat");

  setupSlider(dynamicsSlider_, dynamicsLabel_, "Dynamics");
  dynamicsSlider_.setRange(0.0, 1.0, 0.01);
  dynamicsSlider_.setTextValueSuffix(" (soft → loud)");
  dynamicsSlider_.setDescription(
      "Dynamic range: soft/quiet to loud/intense velocities");

  setupSlider(barsSlider_, barsLabel_, "Bars");
  barsSlider_.setRange(4, 32, 1);
  barsSlider_.setTextValueSuffix(" bars");
  barsSlider_.setDescription("Number of bars to generate: 4 to 32");
  barsSlider_.setNumDecimalPlacesToDisplay(0);

  techniqueLabel_.setFont(juce::Font(13.0f));
  techniqueLabel_.setJustificationType(juce::Justification::centredRight);
  techniqueLabel_.attachToComponent(&techniqueSelector_, true);
  techniqueLabel_.setAccessible(true);
  addAndMakeVisible(techniqueLabel_);
  techniqueSelector_.setTitle("Technique");
  techniqueSelector_.setDescription(
      "High-level articulation or generation technique");
  techniqueSelector_.addItem("None", 1);
  techniqueSelector_.addItem("Legato", 2);
  techniqueSelector_.addItem("Staccato", 3);
  techniqueSelector_.addItem("Arpeggio", 4);
  techniqueSelector_.addItem("Sustain", 5);
  techniqueSelector_.addItem("Muted", 6);
  techniqueSelector_.addItem("Swells", 7);
  addAndMakeVisible(techniqueSelector_);

  songSectionLabel_.setFont(juce::Font(13.0f));
  songSectionLabel_.setJustificationType(juce::Justification::centredRight);
  songSectionLabel_.attachToComponent(&songSectionSelector_, true);
  songSectionLabel_.setAccessible(true);
  addAndMakeVisible(songSectionLabel_);
  songSectionSelector_.setTitle("Song Section");
  songSectionSelector_.setDescription(
      "Which part of the song you are working on");
  songSectionSelector_.addItem("Intro", 1);
  songSectionSelector_.addItem("Verse", 2);
  songSectionSelector_.addItem("Pre-Chorus", 3);
  songSectionSelector_.addItem("Chorus", 4);
  songSectionSelector_.addItem("Bridge", 5);
  songSectionSelector_.addItem("Outro", 6);
  songSectionSelector_.addItem("Full Song", 7);
  addAndMakeVisible(songSectionSelector_);

  setupSlider(songAmountSlider_, songAmountLabel_, "Song Amount");
  songAmountSlider_.setRange(0.0, 100.0, 1.0);
  songAmountSlider_.setTextValueSuffix("%");
  songAmountSlider_.setDescription(
      "How much of the song this generation should cover");
  songAmountSlider_.setNumDecimalPlacesToDisplay(0);

  useHostTempoButton_.setClickingTogglesState(true);
  useHostTempoButton_.setTitle("Use Host Tempo");
  useHostTempoButton_.setDescription(
      "Use the DAW tempo for generation (overrides panel tempo)");
  addAndMakeVisible(useHostTempoButton_);

  autoGenerateButton_.setClickingTogglesState(true);
  autoGenerateButton_.setToggleState(true, juce::dontSendNotification);
  autoGenerateButton_.setTitle("Auto-Generate (Idle)");
  autoGenerateButton_.setDescription(
      "When enabled, changes trigger generation after a short idle delay");
  autoGenerateButton_.onClick = [this]() {
    if (onAutoGenerateToggled) {
      onAutoGenerateToggled(autoGenerateButton_.getToggleState());
    }
  };
  addAndMakeVisible(autoGenerateButton_);

  bypassButton_.setButtonText("Bypass");
  bypassButton_.setClickingTogglesState(true);
  bypassButton_.setTitle("Bypass");
  bypassButton_.setDescription(
      "Bypass Kelly MIDI generation and pass through input MIDI");
  addAndMakeVisible(bypassButton_);

  // Create APVTS attachments (using lowercase IDs to match PluginProcessor)
  valenceAttachment_ =
      std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
          apvts_, PluginProcessor::PARAM_VALENCE, valenceSlider_);
  arousalAttachment_ =
      std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
          apvts_, PluginProcessor::PARAM_AROUSAL, arousalSlider_);
  intensityAttachment_ =
      std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
          apvts_, PluginProcessor::PARAM_INTENSITY, intensitySlider_);
  complexityAttachment_ =
      std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
          apvts_, PluginProcessor::PARAM_COMPLEXITY, complexitySlider_);
  humanizeAttachment_ =
      std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
          apvts_, PluginProcessor::PARAM_HUMANIZE, humanizeSlider_);
  feelAttachment_ =
      std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
          apvts_, PluginProcessor::PARAM_FEEL, feelSlider_);
  dynamicsAttachment_ =
      std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
          apvts_, PluginProcessor::PARAM_DYNAMICS, dynamicsSlider_);
  barsAttachment_ =
      std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
          apvts_, PluginProcessor::PARAM_BARS, barsSlider_);
  techniqueAttachment_ =
      std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
          apvts_, PluginProcessor::PARAM_TECHNIQUE, techniqueSelector_);
  songSectionAttachment_ =
      std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
          apvts_, PluginProcessor::PARAM_SONG_SECTION, songSectionSelector_);
  songAmountAttachment_ =
      std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
          apvts_, PluginProcessor::PARAM_SONG_AMOUNT, songAmountSlider_);
  useHostTempoAttachment_ =
      std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(
          apvts_, PluginProcessor::PARAM_USE_HOST_TEMPO, useHostTempoButton_);
  bypassAttachment_ =
      std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(
          apvts_, PluginProcessor::PARAM_BYPASS, bypassButton_);

  // ========================================================================
  // DISPLAY COMPONENTS
  // ========================================================================
  chordDisplay_.setTitle("Chord Display");
  chordDisplay_.setDescription("Current chord name and notes");
  addAndMakeVisible(chordDisplay_);

  musicTheoryPanel_.setTitle("Music Theory Panel");
  musicTheoryPanel_.setDescription(
      "Music theory settings: key, mode, tempo, instruments");
  addAndMakeVisible(musicTheoryPanel_);

  pianoRollPreview_.setTitle("Piano Roll Preview");
  pianoRollPreview_.setDescription("Preview of generated MIDI notes");
  addAndMakeVisible(pianoRollPreview_);

  // ========================================================================
  // ACTIONS
  // ========================================================================
  generateButton_.setButtonText("Generate");
  generateButton_.onClick = [this]() {
    if (onGenerateClicked) {
      onGenerateClicked();
    }
  };
  generateButton_.setTitle("Generate");
  generateButton_.setDescription(
      "Generate MIDI from current emotional parameters");
  addAndMakeVisible(generateButton_);

  previewButton_.setButtonText("Preview");
  previewButton_.onClick = [this]() {
    if (onPreviewClicked) {
      onPreviewClicked();
    }
  };
  previewButton_.setTitle("Preview");
  previewButton_.setDescription("Preview generated MIDI without exporting");
  addAndMakeVisible(previewButton_);

  exportButton_.setButtonText("Export to DAW");
  exportButton_.onClick = [this]() {
    if (onExportClicked) {
      onExportClicked();
    }
  };
  exportButton_.setTitle("Export to DAW");
  exportButton_.setDescription(
      "Export generated MIDI to your digital audio workstation");
  addAndMakeVisible(exportButton_);

  // ========================================================================
  // PROJECT MENU
  // ========================================================================
  setupProjectMenu();
  projectMenuButton_.setButtonText("Project");
  projectMenuButton_.onClick = [this]() { showProjectMenu(); };
  projectMenuButton_.setTitle("Project Menu");
  projectMenuButton_.setDescription(
      "Project management: New, Open, Save, Save As");
  addAndMakeVisible(projectMenuButton_);

  // ========================================================================
  // VOCAL COMPONENTS
  // ========================================================================
  lyricDisplay_.setTitle("Lyrics");
  lyricDisplay_.setDescription(
      "Display generated lyrics with syllable breakdown");
  addAndMakeVisible(lyricDisplay_);

  vocalControlPanel_.onVoiceTypeChanged = [this](VoiceType type) {
    // Handle voice type change (can be connected to VoiceSynthesizer)
  };
  vocalControlPanel_.onExpressionChanged = [this](const VocalExpression &expr) {
    // Handle expression change (can be connected to VoiceSynthesizer)
  };
  addAndMakeVisible(vocalControlPanel_);

  // Start timer for visualization updates
  startTimer(30); // 30ms = ~33 FPS
}

void EmotionWorkstation::setupSlider(juce::Slider &slider, juce::Label &label,
                                     const juce::String &labelText) {
  slider.setSliderStyle(juce::Slider::LinearHorizontal);
  slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 70, 20);
  slider.setPopupDisplayEnabled(true, true, this);
  slider.setTitle(labelText);
  addAndMakeVisible(slider);

  label.setText(labelText, juce::dontSendNotification);
  label.setFont(juce::Font(13.0f));
  label.setJustificationType(juce::Justification::centredRight);
  label.attachToComponent(&slider, true);
  label.setAccessible(true);
  addAndMakeVisible(label);
}

void EmotionWorkstation::setupButton(juce::Button &button,
                                     const juce::String &tooltip) {
  button.setTooltip(tooltip);
  addAndMakeVisible(button);
}

void EmotionWorkstation::handleEmotionWheelSelection(
    const EmotionNode &emotion) {
  // Update sliders to match selected emotion
  valenceSlider_.setValue(emotion.valence, juce::sendNotificationAsync);
  arousalSlider_.setValue(emotion.arousal, juce::sendNotificationAsync);
  intensitySlider_.setValue(emotion.intensity, juce::sendNotificationAsync);

  // Update radar visualization
  emotionRadar_.setEmotion(emotion.valence, emotion.arousal, emotion.intensity);

  // Notify parent
  if (onEmotionSelected) {
    onEmotionSelected(emotion);
  }
}

void EmotionWorkstation::applyEmotionSelection(const EmotionNode &emotion) {
  emotionWheel_.setSelectedEmotion(emotion.id);
  handleEmotionWheelSelection(emotion);
}

void EmotionWorkstation::paint(juce::Graphics &g) {
  // Background
  g.fillAll(lookAndFeel_.backgroundDark);

  // Section headers
  g.setFont(juce::Font(14.0f, juce::Font::bold));
  g.setColour(lookAndFeel_.textPrimary);

  auto bounds = getLocalBounds();
  int woundHeight = 150;
  int topSectionHeight = 420;
  int theoryHeight = 200;
  int lyricHeight = 150;
  int vocalControlHeight = 200;
  int pianoRollHeight = 150;

  // "EMOTION MAPPING" header
  g.drawText("EMOTION MAPPING", bounds.getX() + 10,
             bounds.getY() + woundHeight + 10, bounds.getWidth() / 2 - 20, 20,
             juce::Justification::centredLeft);

  // "MUSICAL PARAMETERS" header
  g.drawText("MUSICAL PARAMETERS", bounds.getX() + bounds.getWidth() / 2 + 10,
             bounds.getY() + woundHeight + 10, bounds.getWidth() / 2 - 20, 20,
             juce::Justification::centredLeft);

  // Draw section dividers
  g.setColour(lookAndFeel_.borderColor);
  g.drawHorizontalLine(woundHeight, 0.0f, (float)getWidth());
  g.drawHorizontalLine(woundHeight + topSectionHeight, 0.0f, (float)getWidth());
  g.drawHorizontalLine(woundHeight + topSectionHeight + theoryHeight, 0.0f,
                       (float)getWidth());
  g.drawHorizontalLine(woundHeight + topSectionHeight + theoryHeight +
                           lyricHeight,
                       0.0f, (float)getWidth());
  g.drawHorizontalLine(woundHeight + topSectionHeight + theoryHeight +
                           lyricHeight + vocalControlHeight,
                       0.0f, (float)getWidth());
  g.drawHorizontalLine(woundHeight + topSectionHeight + theoryHeight +
                           lyricHeight + vocalControlHeight + pianoRollHeight,
                       0.0f, (float)getWidth());
  g.drawVerticalLine(getWidth() / 2, (float)woundHeight,
                     (float)(woundHeight + topSectionHeight));
}

void EmotionWorkstation::resized() {
  auto bounds = getLocalBounds();
  const int margin = 10;
  const int labelWidth = 80;

  // ========================================================================
  // WOUND INPUT + EMOTION/PREVIEW (TOP)
  // ========================================================================
  const int woundHeight = 150;
  auto woundArea = bounds.removeFromTop(woundHeight).reduced(margin);
  auto woundLabelArea = woundArea.removeFromTop(20);
  woundLabel_.setBounds(woundLabelArea);
  woundArea.removeFromTop(5); // Gap
  auto woundTextArea = woundArea.removeFromTop(55);
  woundInput_.setBounds(woundTextArea);
  woundArea.removeFromTop(8);

  auto emotionPreviewArea = woundArea;
  auto emotionArea =
      emotionPreviewArea.removeFromLeft(emotionPreviewArea.getWidth() / 3);
  auto emotionLabelArea = emotionArea.removeFromTop(18);
  emotionLabel_.setBounds(emotionLabelArea);
  emotionInput_.setBounds(emotionArea);

  emotionPreviewArea.removeFromLeft(margin);
  auto previewLabelArea = emotionPreviewArea.removeFromTop(18);
  previewLabel_.setBounds(previewLabelArea);
  previewText_.setBounds(emotionPreviewArea);

  bounds.removeFromTop(margin); // Gap after wound input

  // ========================================================================
  // MAIN CONTENT AREA (SPLIT LEFT/RIGHT)
  // ========================================================================
  const int topSectionHeight = 420;
  auto topSection = bounds.removeFromTop(topSectionHeight);

  // LEFT: EMOTION MAPPING
  auto leftPanel = topSection.removeFromLeft(getWidth() / 2).reduced(margin);
  leftPanel.removeFromTop(30); // Space for "EMOTION MAPPING" header

  auto emotionWheelArea =
      leftPanel.removeFromTop(leftPanel.getHeight() / 2).reduced(5);
  emotionWheel_.setBounds(emotionWheelArea);

  leftPanel.removeFromTop(10); // Gap
  auto emotionRadarArea = leftPanel.reduced(5);
  emotionRadar_.setBounds(emotionRadarArea);

  // RIGHT: MUSICAL PARAMETERS
  auto rightPanel = topSection.reduced(margin);
  rightPanel.removeFromTop(30); // Space for "MUSICAL PARAMETERS" header

  const int sliderHeight = 30;
  const int sliderGap = 5;

  valenceSlider_.setBounds(rightPanel.removeFromTop(sliderHeight));
  rightPanel.removeFromTop(sliderGap);

  arousalSlider_.setBounds(rightPanel.removeFromTop(sliderHeight));
  rightPanel.removeFromTop(sliderGap);

  intensitySlider_.setBounds(rightPanel.removeFromTop(sliderHeight));
  rightPanel.removeFromTop(sliderGap);

  complexitySlider_.setBounds(rightPanel.removeFromTop(sliderHeight));
  rightPanel.removeFromTop(sliderGap);

  humanizeSlider_.setBounds(rightPanel.removeFromTop(sliderHeight));
  rightPanel.removeFromTop(sliderGap);

  feelSlider_.setBounds(rightPanel.removeFromTop(sliderHeight));
  rightPanel.removeFromTop(sliderGap);

  dynamicsSlider_.setBounds(rightPanel.removeFromTop(sliderHeight));
  rightPanel.removeFromTop(sliderGap);

  barsSlider_.setBounds(rightPanel.removeFromTop(sliderHeight));
  rightPanel.removeFromTop(sliderGap);

  techniqueSelector_.setBounds(rightPanel.removeFromTop(sliderHeight));
  rightPanel.removeFromTop(sliderGap);

  songSectionSelector_.setBounds(rightPanel.removeFromTop(sliderHeight));
  rightPanel.removeFromTop(sliderGap);

  songAmountSlider_.setBounds(rightPanel.removeFromTop(sliderHeight));
  rightPanel.removeFromTop(sliderGap);

  auto hostTempoArea = rightPanel.removeFromTop(24).reduced(labelWidth, 0);
  useHostTempoButton_.setBounds(hostTempoArea);
  rightPanel.removeFromTop(sliderGap);

  auto autoGenArea = rightPanel.removeFromTop(24).reduced(labelWidth, 0);
  autoGenerateButton_.setBounds(autoGenArea);
  rightPanel.removeFromTop(sliderGap);

  // Bypass button below sliders
  auto bypassArea = rightPanel.removeFromTop(24).reduced(labelWidth, 0);
  bypassButton_.setBounds(bypassArea);

  bounds.removeFromTop(margin); // Gap

  // ========================================================================
  // THEORY & DISPLAY ROW
  // ========================================================================
  const int theoryHeight = 200;
  auto theoryRow = bounds.removeFromTop(theoryHeight).reduced(margin);

  auto chordDisplayArea = theoryRow.removeFromLeft(150);
  chordDisplay_.setBounds(chordDisplayArea);

  theoryRow.removeFromLeft(margin); // Gap
  musicTheoryPanel_.setBounds(theoryRow);

  bounds.removeFromTop(margin); // Gap

  // ========================================================================
  // LYRIC DISPLAY
  // ========================================================================
  const int lyricHeight = 150;
  auto lyricArea = bounds.removeFromTop(lyricHeight).reduced(margin);
  lyricDisplay_.setBounds(lyricArea);

  bounds.removeFromTop(margin); // Gap

  // ========================================================================
  // VOCAL CONTROLS
  // ========================================================================
  const int vocalControlHeight = 200;
  auto vocalControlArea =
      bounds.removeFromTop(vocalControlHeight).reduced(margin);
  vocalControlPanel_.setBounds(vocalControlArea);

  bounds.removeFromTop(margin); // Gap

  // ========================================================================
  // PIANO ROLL PREVIEW
  // ========================================================================
  const int pianoRollHeight = 150;
  auto pianoRollArea = bounds.removeFromTop(pianoRollHeight).reduced(margin);
  pianoRollPreview_.setBounds(pianoRollArea);

  bounds.removeFromTop(margin); // Gap

  // ========================================================================
  // ACTIONS (BOTTOM)
  // ========================================================================
  auto actionsArea = bounds.reduced(margin);
  const int buttonWidth = 120;
  const int buttonHeight = 40;
  const int buttonGap = 10;

  auto actionsRow = actionsArea.removeFromTop(buttonHeight);
  actionsRow = actionsRow.withSizeKeepingCentre(buttonWidth * 5 + buttonGap * 4,
                                                buttonHeight);

  projectMenuButton_.setBounds(actionsRow.removeFromLeft(buttonWidth));
  actionsRow.removeFromLeft(buttonGap);

  generateButton_.setBounds(actionsRow.removeFromLeft(buttonWidth));
  actionsRow.removeFromLeft(buttonGap);

  previewButton_.setBounds(actionsRow.removeFromLeft(buttonWidth));
  actionsRow.removeFromLeft(buttonGap);

  exportButton_.setBounds(actionsRow.removeFromLeft(buttonWidth));
  actionsRow.removeFromLeft(buttonGap);

  // Bypass button is now in the right panel above
}

void EmotionWorkstation::timerCallback() {
  // Update visualizations based on current parameter values
  float valence = static_cast<float>(valenceSlider_.getValue());
  float arousal = static_cast<float>(arousalSlider_.getValue());
  float intensity = static_cast<float>(intensitySlider_.getValue());

  emotionRadar_.setEmotion(valence, arousal, intensity);
}

void EmotionWorkstation::setupProjectMenu() {
  projectMenu_.clear();
  projectMenu_.addItem(1, "New Project", true, false);
  projectMenu_.addItem(2, "Open Project...", true, false);
  projectMenu_.addSeparator();
  projectMenu_.addItem(3, "Save Project", true, false);
  projectMenu_.addItem(4, "Save Project As...", true, false);
}

void EmotionWorkstation::showProjectMenu() {
  setupProjectMenu();

  projectMenu_.showMenuAsync(juce::PopupMenu::Options()
                                 .withTargetComponent(&projectMenuButton_)
                                 .withParentComponent(getTopLevelComponent()),
                             [this](int result) {
                               switch (result) {
                               case 1: // New Project
                                 if (onNewProject) {
                                   onNewProject();
                                 }
                                 break;
                               case 2: // Open Project
                                 if (onOpenProject) {
                                   onOpenProject();
                                 }
                                 break;
                               case 3: // Save Project
                                 if (onSaveProject) {
                                   onSaveProject();
                                 }
                                 break;
                               case 4: // Save Project As
                                 if (onSaveProjectAs) {
                                   onSaveProjectAs();
                                 }
                                 break;
                               }
                             });
}

} // namespace kelly
