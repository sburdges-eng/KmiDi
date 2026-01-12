#include "ui/ScoreEntryPanel.h"
#include <juce_graphics/juce_graphics.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include <algorithm>

namespace midikompanion
{

ScoreEntryPanel::ScoreEntryPanel()
    : entryMode_(EntryMode::Standard),
      viewMode_(ViewMode::SingleStaff),
      currentClef_(Clef::Treble),
      currentNoteValue_(NoteValue::Quarter),
      currentDynamic_(Dynamic::MezzoForte),
      currentArticulation_(Articulation::None),
      cursorMeasure_(1),
      cursorBeat_(1.0f),
      showChordSymbols_(true),
      showLyrics_(true),
      showDynamics_(true),
      zoomFactor_(1.0f)
{
    initializeComponents();
    initializeTemplates();

    // Default time and key signatures
    setTimeSignature(4, 4);
    setKeySignature("C Major");
    setTempo(120.0f, "Allegro");

    // Register for callbacks (example)
    // playButton_->onClick = [this] { onPlayClicked(); };
}

void ScoreEntryPanel::initializeComponents()
{
    // --- Entry Mode Selector ---
    entryModeSelector_.reset(new juce::ComboBox("Entry Mode"));
    entryModeSelector_->addItem("Simple", (int)EntryMode::Simple + 1);
    entryModeSelector_->addItem("Standard", (int)EntryMode::Standard + 1);
    entryModeSelector_->addItem("Professional", (int)EntryMode::Professional + 1);
    entryModeSelector_->addItem("Chord", (int)EntryMode::Chord + 1);
    entryModeSelector_->setSelectedId((int)entryMode_ + 1);
    entryModeSelector_->onChange = [this] {
        setEntryMode((EntryMode)(entryModeSelector_->getSelectedId() - 1));
    };
    addAndMakeVisible(*entryModeSelector_);

    // --- Score Display Area ---
    scoreDisplay_.reset(new juce::Component());
    scoreViewport_.reset(new juce::Viewport("Score Viewport"));
    scoreViewport_->setViewedComponent(scoreDisplay_.get());
    addAndMakeVisible(*scoreViewport_);

    // --- Note Value Buttons ---
    wholeNoteButton_.reset(new juce::TextButton("Whole"));
    wholeNoteButton_->onClick = [this] { onNoteValueSelected(NoteValue::Whole); };
    addAndMakeVisible(*wholeNoteButton_);
    halfNoteButton_.reset(new juce::TextButton("Half"));
    halfNoteButton_->onClick = [this] { onNoteValueSelected(NoteValue::Half); };
    addAndMakeVisible(*halfNoteButton_);
    quarterNoteButton_.reset(new juce::TextButton("Quarter"));
    quarterNoteButton_->onClick = [this] { onNoteValueSelected(NoteValue::Quarter); };
    addAndMakeVisible(*quarterNoteButton_);
    eighthNoteButton_.reset(new juce::TextButton("Eighth"));
    eighthNoteButton_->onClick = [this] { onNoteValueSelected(NoteValue::Eighth); };
    addAndMakeVisible(*eighthNoteButton_);
    sixteenthNoteButton_.reset(new juce::TextButton("16th"));
    sixteenthNoteButton_->onClick = [this] { onNoteValueSelected(NoteValue::Sixteenth); };
    addAndMakeVisible(*sixteenthNoteButton_);
    dottedButton_.reset(new juce::TextButton("Dot"));
    dottedButton_->onClick = [this] { toggleDot(); };
    addAndMakeVisible(*dottedButton_);
    tripletButton_.reset(new juce::TextButton("Triplet"));
    tripletButton_->onClick = [this] { makeTriplet(); };
    addAndMakeVisible(*tripletButton_);

    // --- Dynamics Buttons ---
    ppButton_.reset(new juce::TextButton("pp"));
    ppButton_->onClick = [this] { onDynamicSelected(Dynamic::Pianissimo); };
    addAndMakeVisible(*ppButton_);
    pButton_.reset(new juce::TextButton("p"));
    pButton_->onClick = [this] { onDynamicSelected(Dynamic::Piano); };
    addAndMakeVisible(*pButton_);
    mpButton_.reset(new juce::TextButton("mp"));
    mpButton_->onClick = [this] { onDynamicSelected(Dynamic::MezzoPiano); };
    addAndMakeVisible(*mpButton_);
    mfButton_.reset(new juce::TextButton("mf"));
    mfButton_->onClick = [this] { onDynamicSelected(Dynamic::MezzoForte); };
    addAndMakeVisible(*mfButton_);
    fButton_.reset(new juce::TextButton("f"));
    fButton_->onClick = [this] { onDynamicSelected(Dynamic::Forte); };
    addAndMakeVisible(*fButton_);
    ffButton_.reset(new juce::TextButton("ff"));
    ffButton_->onClick = [this] { onDynamicSelected(Dynamic::Fortissimo); };
    addAndMakeVisible(*ffButton_);

    // --- Quick Entry ---
    quickEntryInput_.reset(new juce::TextEditor("Quick Entry"));
    quickEntryInput_->setText("C major scale quarter notes");
    addAndMakeVisible(*quickEntryInput_);
    quickEntryButton_.reset(new juce::TextButton("Go"));
    quickEntryButton_->onClick = [this] { onQuickEntryExecuted(); };
    addAndMakeVisible(*quickEntryButton_);

    // --- Playback Controls ---
    playButton_.reset(new juce::TextButton("Play"));
    playButton_->onClick = [this] { onPlayClicked(); };
    addAndMakeVisible(*playButton_);
    stopButton_.reset(new juce::TextButton("Stop"));
    stopButton_->onClick = [this] { onStopClicked(); };
    addAndMakeVisible(*stopButton_);
    metronomeButton_.reset(new juce::TextButton("Metronome"));
    metronomeButton_->onClick = [this] { onMetronomeToggled(); };
    addAndMakeVisible(*metronomeButton_);

    // --- Time/Key/Tempo Labels ---
    timeSignatureLabel_.reset(new juce::Label("Time Sig", "4/4"));
    addAndMakeVisible(*timeSignatureLabel_);
    keySignatureLabel_.reset(new juce::Label("Key Sig", "C Major"));
    addAndMakeVisible(*keySignatureLabel_);
    tempoLabel_.reset(new juce::Label("Tempo", "120 BPM"));
    addAndMakeVisible(*tempoLabel_);

    // --- Music Theory Brain ---
    theoryBrain_.reset(new theory::MusicTheoryBrain());
}

void ScoreEntryPanel::initializeTemplates()
{
    // Example templates
    ScoreTemplate cMajorScale;
    cMajorScale.name = "C Major Scale";
    cMajorScale.description = "Ascending C Major scale in quarter notes.";
    // Populate notes for C Major Scale
    for (int pitch = 60; pitch <= 72; ++pitch) // C4 to C5
    {
        cMajorScale.notes.push_back({pitch, NoteValue::Quarter, false, false, Dynamic::MezzoForte, Articulation::None, false, "", 1, 1.0f});
    }
    templates_.push_back(cMajorScale);

    ScoreTemplate bluesProgression;
    bluesProgression.name = "12-Bar Blues";
    bluesProgression.description = "Standard 12-bar blues chord progression in C.";
    bluesProgression.chords.push_back({"C7", 1, 1.0f});
    bluesProgression.chords.push_back({"F7", 5, 1.0f});
    bluesProgression.chords.push_back({"C7", 7, 1.0f});
    bluesProgression.chords.push_back({"G7", 9, 1.0f});
    bluesProgression.chords.push_back({"F7", 10, 1.0f});
    bluesProgression.chords.push_back({"C7", 11, 1.0f});
    templates_.push_back(bluesProgression);
}

void ScoreEntryPanel::paint(juce::Graphics& g) {
    g.fillAll(juce::Colour(0xff2a2a2a)); // Dark background

    juce::Rectangle<int> scoreArea = getScoreArea();
    drawStaff(g, scoreArea); // Draw staff lines
    drawKeySignature(g, keySignatures_.empty() ? "C Major" : keySignatures_.back().key, scoreArea.getTopLeft().withX(scoreArea.getX() + 20));
    drawTimeSignature(g, timeSignatures_.empty() ? 4 : timeSignatures_.back().numerator,
                      timeSignatures_.empty() ? 4 : timeSignatures_.back().denominator,
                      scoreArea.getTopLeft().withX(scoreArea.getX() + 60));

    // Draw notes, chords, lyrics etc.
    // (Simplified drawing logic)
    float currentX = scoreArea.getX();
    int currentMeasure = 1;
    float beatsPerMeasure = timeSignatures_.empty() ? 4.0f : timeSignatures_.back().numerator;
    float pixelsPerBeat = scoreArea.getWidth() / (beatsPerMeasure * 4.0f); // Assuming 4 bars visible

    for (const auto& note : notes_) {
        // Convert note.beat to pixel position
        float noteX = currentX + (note.beat - 1.0f) * pixelsPerBeat;
        drawNote(g, note, juce::Point<int>(static_cast<int>(noteX), 100)); // Placeholder Y
    }
    for (const auto& chord : chordSymbols_) {
        // Draw chord symbols above staff
        float chordX = currentX + (chord.beat - 1.0f) * pixelsPerBeat;
        drawChordSymbol(g, chord, juce::Point<int>(static_cast<int>(chordX), 50)); // Placeholder Y
    }

    drawCursor(g); // Draw input cursor
}

void ScoreEntryPanel::resized() {
    juce::Rectangle<int> bounds = getLocalBounds();

    // Layout components
    int y = 0;
    entryModeSelector_->setBounds(10, y + 10, 150, 24);
    keySignatureLabel_->setBounds(bounds.getWidth() - 250, y + 10, 100, 24);
    timeSignatureLabel_->setBounds(bounds.getWidth() - 150, y + 10, 50, 24);
    tempoLabel_->setBounds(bounds.getWidth() - 90, y + 10, 80, 24);
    y += 40;

    scoreViewport_->setBounds(10, y + 10, bounds.getWidth() - 20, bounds.getHeight() - y - 100);

    int buttonWidth = 60;
    int buttonHeight = 30;
    int x = 10;
    y = bounds.getHeight() - 80;

    wholeNoteButton_->setBounds(x, y, buttonWidth, buttonHeight); x += buttonWidth + 5;
    halfNoteButton_->setBounds(x, y, buttonWidth, buttonHeight); x += buttonWidth + 5;
    quarterNoteButton_->setBounds(x, y, buttonWidth, buttonHeight); x += buttonWidth + 5;
    eighthNoteButton_->setBounds(x, y, buttonWidth, buttonHeight); x += buttonWidth + 5;
    sixteenthNoteButton_->setBounds(x, y, buttonWidth, buttonHeight); x += buttonWidth + 5;
    dottedButton_->setBounds(x, y, buttonWidth, buttonHeight); x += buttonWidth + 5;
    tripletButton_->setBounds(x, y, buttonWidth, buttonHeight); x += buttonWidth + 5;

    x = 10;
    y += buttonHeight + 5;
    ppButton_->setBounds(x, y, 40, buttonHeight); x += 45;
    pButton_->setBounds(x, y, 40, buttonHeight); x += 45;
    mpButton_->setBounds(x, y, 40, buttonHeight); x += 45;
    mfButton_->setBounds(x, y, 40, buttonHeight); x += 45;
    fButton_->setBounds(x, y, 40, buttonHeight); x += 45;
    ffButton_->setBounds(x, y, 40, buttonHeight); x += 45;

    x = bounds.getWidth() - 250;
    playButton_->setBounds(x, y, 60, buttonHeight); x += 65;
    stopButton_->setBounds(x, y, 60, buttonHeight); x += 65;
    metronomeButton_->setBounds(x, y, 90, buttonHeight); x += 95;

    quickEntryInput_->setBounds(10, bounds.getHeight() - 40, bounds.getWidth() - 150, 24);
    quickEntryButton_->setBounds(bounds.getWidth() - 135, bounds.getHeight() - 40, 120, 24);
}

void ScoreEntryPanel::mouseDown(const juce::MouseEvent& event) {
    if (scoreDisplay_ && event.originalComponent == scoreDisplay_.get()) {
        // Map mouse click to a musical position
        juce::Point<int> posInScore = event.getPosition() - scoreViewport_->getViewPosition();
        int pitch = staffPositionToPitch(posInScore.getY(), currentClef_);
        float beat = (float)posInScore.getX() / (getScoreArea().getWidth() / (timeSignatures_.empty() ? 4.0f : timeSignatures_.back().numerator * 4.0f)); // Simplified

        // Add a note at this position (example)
        addNote(pitch, currentNoteValue_, dottedButton_->getToggleState());
    }
}

void ScoreEntryPanel::mouseDrag(const juce::MouseEvent& event) {
    // Implement dragging for selection or note length adjustment
}

void ScoreEntryPanel::setEntryMode(EntryMode mode) {
    entryMode_ = mode;
    // Update UI elements based on mode
    // repaint();
}

void ScoreEntryPanel::setTimeSignature(int numerator, int denominator) {
    timeSignatures_.push_back({numerator, denominator, getMaxMeasureIndex() + 1});
    timeSignatureLabel_->setText(juce::String(numerator) + "/" + juce::String(denominator), juce::dontSendNotification);
    repaint();
}

void ScoreEntryPanel::setKeySignature(const std::string& key) {
    keySignatures_.push_back({key, getMaxMeasureIndex() + 1});
    keySignatureLabel_->setText(key, juce::dontSendNotification);
    repaint();
}

void ScoreEntryPanel::setTempo(float bpm, const std::string& description) {
    tempos_.push_back({bpm, description, getMaxMeasureIndex() + 1});
    tempoLabel_->setText(juce::String(bpm) + " BPM", juce::dontSendNotification);
    repaint();
}

void ScoreEntryPanel::setClef(Clef clef) {
    currentClef_ = clef;
    repaint();
}

void ScoreEntryPanel::addNote(int pitch, NoteValue duration, bool dotted) {
    // Create a new note at the current cursor position
    NotationNote newNote = {pitch, duration, dotted, false, currentDynamic_, currentArticulation_, false, "", cursorMeasure_, cursorBeat_};
    notes_.push_back(newNote);
    // Advance cursor
    cursorBeat_ += 1.0f; // For a quarter note
    if (cursorBeat_ > (timeSignatures_.empty() ? 4.0f : timeSignatures_.back().numerator)) {
        cursorBeat_ = 1.0f;
        cursorMeasure_++;
    }
    repaint();
}

void ScoreEntryPanel::addChord(const std::vector<int>& pitches, NoteValue duration) {
    // Add multiple notes at the same cursor position
    for (int pitch : pitches) {
        NotationNote newNote = {pitch, duration, false, false, currentDynamic_, currentArticulation_, false, "", cursorMeasure_, cursorBeat_};
        notes_.push_back(newNote);
    }
    // Advance cursor once for the chord
    cursorBeat_ += 1.0f; // For a quarter note
    if (cursorBeat_ > (timeSignatures_.empty() ? 4.0f : timeSignatures_.back().numerator)) {
        cursorBeat_ = 1.0f;
        cursorMeasure_++;
    }
    repaint();
}

void ScoreEntryPanel::addChordSymbol(const std::string& symbol) {
    chordSymbols_.push_back({symbol, cursorMeasure_, cursorBeat_});
    repaint();
}

void ScoreEntryPanel::addRest(NoteValue duration) {
    // Placeholder for adding rests
    // Advance cursor based on rest duration
    cursorBeat_ += 1.0f; // For a quarter rest
    if (cursorBeat_ > (timeSignatures_.empty() ? 4.0f : timeSignatures_.back().numerator)) {
        cursorBeat_ = 1.0f;
        cursorMeasure_++;
    }
    repaint();
}

void ScoreEntryPanel::setDynamic(Dynamic dynamic) {
    currentDynamic_ = dynamic;
    // Apply to selected notes or next entered notes
}

void ScoreEntryPanel::setArticulation(Articulation articulation) {
    currentArticulation_ = articulation;
    // Apply to selected notes or next entered notes
}

void ScoreEntryPanel::addLyric(const std::string& text) {
    // Add lyric to nearest note
}

void ScoreEntryPanel::toggleDot() {
    currentNoteValue_ = (currentNoteValue_ == NoteValue::Dotted) ? NoteValue::Quarter : NoteValue::Dotted; // Simple toggle for example
}

void ScoreEntryPanel::makeTriplet() {
    // Convert selected notes to triplet
}

void ScoreEntryPanel::toggleTie() {
    // Toggle tie for selected note
}

void ScoreEntryPanel::moveCursorForward() {
    cursorBeat_ += 1.0f;
    if (cursorBeat_ > (timeSignatures_.empty() ? 4.0f : timeSignatures_.back().numerator)) {
        cursorBeat_ = 1.0f;
        cursorMeasure_++;
    }
    repaint();
}

void ScoreEntryPanel::moveCursorBackward() {
    cursorBeat_ -= 1.0f;
    if (cursorBeat_ < 1.0f) {
        cursorMeasure_ = std::max(1, cursorMeasure_ - 1);
        cursorBeat_ = (timeSignatures_.empty() ? 4.0f : timeSignatures_.back().numerator);
    }
    repaint();
}

void ScoreEntryPanel::moveCursorToMeasure(int measure) {
    cursorMeasure_ = std::max(1, measure);
    cursorBeat_ = 1.0f;
    repaint();
}

void ScoreEntryPanel::moveCursorToNextMeasure() {
    cursorMeasure_++;
    cursorBeat_ = 1.0f;
    repaint();
}

void ScoreEntryPanel::loadTemplate(const ScoreTemplate& template_) {
    notes_ = template_.notes;
    chordSymbols_ = template_.chords;
    // Reset cursor
    cursorMeasure_ = 1;
    cursorBeat_ = 1.0f;
    repaint();
}

std::vector<ScoreEntryPanel::ScoreTemplate> ScoreEntryPanel::getAvailableTemplates() const {
    return templates_;
}

void ScoreEntryPanel::quickEntry(const std::string& description) {
    parseQuickEntry(description);
    repaint();
}

juce::MidiBuffer ScoreEntryPanel::toMidiBuffer() const {
    juce::MidiBuffer midiBuffer;
    // Convert notes_ to MIDI messages and add to buffer
    // (Complex logic involving tempo, time signature, note values, etc.)
    return midiBuffer;
}

void ScoreEntryPanel::fromMidiBuffer(const juce::MidiBuffer& buffer) {
    notes_.clear();
    chordSymbols_.clear();
    // Convert MIDI messages in buffer to NotationNote objects
    // (Complex logic)
    repaint();
}

void ScoreEntryPanel::playFromStart() {
    if (onPlayRequested) {
        onPlayRequested(toMidiBuffer());
    }
}

void ScoreEntryPanel::playFromCursor() {
    // Generate MIDI from cursor position and play
}

void ScoreEntryPanel::stop() {
    if (onStopRequested) {
        onStopRequested();
    }
}

void ScoreEntryPanel::toggleMetronome() {
    // Toggle metronome state
    if (onMetronomeToggledCallback) {
        // onMetronomeToggledCallback(metronome_enabled_);
    }
}

void ScoreEntryPanel::setViewMode(ViewMode mode) {
    viewMode_ = mode;
    // Adjust display based on view mode
    resized();
    repaint();
}

void ScoreEntryPanel::setZoom(float zoomFactor) {
    zoomFactor_ = juce::jlimit(0.5f, 2.0f, zoomFactor);
    // Adjust scaling of score elements
    repaint();
}

void ScoreEntryPanel::setShowChordSymbols(bool show) {
    showChordSymbols_ = show;
    repaint();
}

void ScoreEntryPanel::setShowLyrics(bool show) {
    showLyrics_ = show;
    repaint();
}

void ScoreEntryPanel::setShowDynamics(bool show) {
    showDynamics_ = show;
    repaint();
}

int ScoreEntryPanel::getMaxMeasureIndex() const {
    if (notes_.empty() && chordSymbols_.empty() && timeSignatures_.empty() && keySignatures_.empty() && tempos_.empty()) {
        return 0;
    }

    int maxMeasure = 0;
    for (const auto& note : notes_) { maxMeasure = std::max(maxMeasure, note.measure); }
    for (const auto& chord : chordSymbols_) { maxMeasure = std::max(maxMeasure, chord.measure); }
    for (const auto& ts : timeSignatures_) { maxMeasure = std::max(maxMeasure, ts.measure); }
    for (const auto& ks : keySignatures_) { maxMeasure = std::max(maxMeasure, ks.measure); }
    for (const auto& tm : tempos_) { maxMeasure = std::max(maxMeasure, tm.measure); }

    return maxMeasure;
}

juce::Rectangle<int> ScoreEntryPanel::getScoreArea() const {
    // Define the main area where the score is drawn
    return getLocalBounds().reduced(20, 100); // Example: 20px padding, 100px from top
}

void ScoreEntryPanel::drawStaff(juce::Graphics& g, juce::Rectangle<int> area) {
    g.setColour(juce::Colours::white); // Staff line color
    float staffTop = (float)area.getY() + (float)area.getHeight() * 0.2f; // Top of staff lines
    float lineSpacing = 10.0f * zoomFactor_; // Adjust spacing with zoom

    for (int i = 0; i < 5; ++i) {
        float y = staffTop + (float)i * lineSpacing;
        g.drawLine(area.getX(), y, area.getRight(), y, 1.0f); // Draw 5 lines
    }

    // Draw ledger lines if necessary for notes outside the staff
}

void ScoreEntryPanel::drawClef(juce::Graphics& g, Clef clef, juce::Point<int> position) {
    // Placeholder: Draw different clef symbols
    g.setColour(juce::Colours::white);
    juce::Font font("Times New Roman", 40.0f * zoomFactor_, juce::Font::plain);
    g.setFont(font);

    juce::String clefSymbol;
    switch (clef) {
        case Clef::Treble: clefSymbol = "& #x1d11e;"; break; // G Clef symbol
        case Clef::Bass:   clefSymbol = "& #x1d122;"; break; // F Clef symbol
        case Clef::Alto:   clefSymbol = "& #x1d121;"; break; // C Clef symbol
        case Clef::Tenor:  clefSymbol = "& #x1d121;"; break; // C Clef symbol (octave down)
        case Clef::Percussion: clefSymbol = "Perc"; break;
    }
    g.drawText(clefSymbol, position.getX(), position.getY(), 50, 50, juce::Justification::centred, false);
}

void ScoreEntryPanel::drawTimeSignature(juce::Graphics& g, int numerator, int denominator, juce::Point<int> position) {
    g.setColour(juce::Colours::white);
    juce::Font font("Times New Roman", 24.0f * zoomFactor_, juce::Font::plain);
    g.setFont(font);

    juce::String numStr = juce::String(numerator);
    juce::String denStr = juce::String(denominator);

    g.drawText(numStr, position.getX(), position.getY() - 10, 30, 30, juce::Justification::centred, false);
    g.drawText(denStr, position.getX(), position.getY() + 10, 30, 30, juce::Justification::centred, false);
}

void ScoreEntryPanel::drawKeySignature(juce::Graphics& g, const std::string& key, juce::Point<int> position) {
    g.setColour(juce::Colours::white);
    juce::Font font("Times New Roman", 20.0f * zoomFactor_, juce::Font::plain);
    g.setFont(font);

    // Simplified: just draw key name
    g.drawText(key, position.getX(), position.getY(), 80, 20, juce::Justification::left, false);
}

void ScoreEntryPanel::drawNote(juce::Graphics& g, const NotationNote& note, juce::Point<int> position) {
    g.setColour(juce::Colours::white);
    // Placeholder: Draw a simple oval for the note head
    g.fillEllipse((float)position.getX(), (float)position.getY(), 15.0f * zoomFactor_, 10.0f * zoomFactor_);
    // Draw stem, flag, etc. based on duration, dotted, triplet
}

void ScoreEntryPanel::drawChordSymbol(juce::Graphics& g, const ChordSymbol& chord, juce::Point<int> position) {
    if (showChordSymbols_) {
        g.setColour(juce::Colours::cyan);
        juce::Font font("Arial", 14.0f * zoomFactor_, juce::Font::bold);
        g.setFont(font);
        g.drawText(chord.symbol, position.getX(), position.getY(), 50, 20, juce::Justification::left, false);
    }
}

void ScoreEntryPanel::drawBarline(juce::Graphics& g, int x) {
    g.setColour(juce::Colours::lightgrey);
    g.drawLine(x, getScoreArea().getY(), x, getScoreArea().getBottom(), 1.0f);
}

void ScoreEntryPanel::drawCursor(juce::Graphics& g) {
    g.setColour(juce::Colours::red.withAlpha(0.6f));
    // Draw a vertical line or small rectangle at cursorMeasure_ / cursorBeat_
    juce::Rectangle<int> scoreArea = getScoreArea();
    float beatsPerMeasure = timeSignatures_.empty() ? 4.0f : timeSignatures_.back().numerator;
    float pixelsPerBeat = scoreArea.getWidth() / (beatsPerMeasure * 4.0f); // Assuming 4 bars visible

    float cursorX = scoreArea.getX() + (cursorBeat_ - 1.0f + (cursorMeasure_ - 1) * beatsPerMeasure) * pixelsPerBeat;
    g.drawLine(cursorX, scoreArea.getY(), cursorX, scoreArea.getBottom(), 2.0f);
}

int ScoreEntryPanel::staffPositionToPitch(int yPosition, Clef clef) const {
    // Simplified: Treble clef, middle C (MIDI 60) is roughly at a certain Y position
    // This needs to be calibrated based on staff drawing logic
    return 60; // Placeholder
}

int ScoreEntryPanel::pitchToStaffPosition(int pitch, Clef clef) const {
    // Simplified: MIDI pitch to Y position
    return 100; // Placeholder
}

void ScoreEntryPanel::parseQuickEntry(const std::string& text) {
    // Placeholder for natural language parsing to notes/chords
    // For example, if text is "C major scale quarter notes":
    // addNote(60, NoteValue::Quarter); addNote(62, NoteValue::Quarter); ...
}

void ScoreEntryPanel::onNoteValueSelected(NoteValue value) {
    currentNoteValue_ = value;
    // Update button states
}

void ScoreEntryPanel::onDynamicSelected(Dynamic dynamic) {
    currentDynamic_ = dynamic;
}

void ScoreEntryPanel::onArticulationSelected(Articulation articulation) {
    currentArticulation_ = articulation;
}

void ScoreEntryPanel::onQuickEntryExecuted() {
    std::string text = quickEntryInput_->getText().toStdString();
    parseQuickEntry(text);
}

void ScoreEntryPanel::onPlayClicked() {
    if (onPlayRequested) {
        onPlayRequested(toMidiBuffer());
    }
}

void ScoreEntryPanel::onStopClicked() {
    if (onStopRequested) {
        onStopRequested();
    }
}

void ScoreEntryPanel::onMetronomeToggled() {
    // Toggle internal metronome state
    // if (onMetronomeToggledCallback) {
    //     onMetronomeToggledCallback(metronome_enabled_);
    // }
}

} // namespace midikompanion
