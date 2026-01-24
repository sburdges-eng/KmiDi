#include "ui/MixerConsolePanel.h"
#include <juce_graphics/juce_graphics.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_basics/juce_audio_basics.h> // For MidiBuffer
#include <algorithm>

namespace midikompanion
{

//==============================================================================
// ChannelStrip Implementation
//==============================================================================

ChannelStrip::ChannelStrip(const std::string& channelName)
    : channelName_(channelName),
      gainDb_(0.0f),
      pan_(0.0f),
      muted_(false),
      soloed_(false),
      recordArmed_(false),
      lowEQ_({100.0f, 0.0f, 0.707f}), // Default EQ settings
      midEQ_({1000.0f, 0.0f, 0.707f}),
      highEQ_({10000.0f, 0.0f, 0.707f}),
      peakLevel_(0.0f),
      rmsLevel_(0.0f)
{
    // Initialize UI components for the channel strip
    gainFader_.reset(new juce::Slider("Gain"));
    gainFader_->setSliderStyle(juce::Slider::LinearVertical);
    gainFader_->setRange(-60.0, 12.0, 0.1); // -60dB to +12dB
    gainFader_->setValue(gainDb_);
    gainFader_->setTextBoxIsEditable(true); // Corrected argument count
    gainFader_->setDoubleClickReturnValue(true, 0.0); // Double-click to reset gain
    gainFader_->onValueChange = [this] {
        setGain(static_cast<float>(gainFader_->getValue()));
        if (onGainChanged) onGainChanged(gainDb_);
    };
    addAndMakeVisible(*gainFader_);

    panKnob_.reset(new juce::Slider("Pan"));
    panKnob_->setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    panKnob_->setRange(-1.0, 1.0, 0.01); // -1.0 (left) to +1.0 (right)
    panKnob_->setValue(pan_);
    panKnob_->setTextBoxIsEditable(true); // Corrected argument count
    panKnob_->setDoubleClickReturnValue(true, 0.0); // Double-click to reset pan
    panKnob_->onValueChange = [this] {
        setPan(static_cast<float>(panKnob_->getValue()));
        if (onPanChanged) onPanChanged(pan_);
    };
    addAndMakeVisible(*panKnob_);

    muteButton_.reset(new juce::TextButton("M"));
    muteButton_->setToggleState(muted_, juce::dontSendNotification);
    muteButton_->onClick = [this] {
        setMute(!muted_);
        if (onMuteChanged) onMuteChanged(muted_);
    };
    addAndMakeVisible(*muteButton_);

    soloButton_.reset(new juce::TextButton("S"));
    soloButton_->setToggleState(soloed_, juce::dontSendNotification);
    soloButton_->onClick = [this] {
        setSolo(!soloed_);
        if (onSoloChanged) onSoloChanged(soloed_);
    };
    addAndMakeVisible(*soloButton_);

    recordArmButton_.reset(new juce::TextButton("R"));
    recordArmButton_->setToggleState(recordArmed_, juce::dontSendNotification);
    recordArmButton_->onClick = [this] {
        setRecordArm(!recordArmed_);
    };
    addAndMakeVisible(*recordArmButton_);

    // EQ knobs (simplified for this example)
    lowEQKnob_.reset(new juce::Slider("Low EQ"));
    lowEQKnob_->setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    lowEQKnob_->setRange(-12.0, 12.0, 0.1); // +/- 12dB
    lowEQKnob_->setValue(lowEQ_.gain);
    lowEQKnob_->onValueChange = [this] {
        setLowEQ(static_cast<float>(lowEQKnob_->getValue()));
    };
    addAndMakeVisible(*lowEQKnob_);

    midEQKnob_.reset(new juce::Slider("Mid EQ"));
    midEQKnob_->setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    midEQKnob_->setRange(-12.0, 12.0, 0.1);
    midEQKnob_->setValue(midEQ_.gain);
    midEQKnob_->onValueChange = [this] {
        setMidEQ(static_cast<float>(midEQKnob_->getValue()));
    };
    addAndMakeVisible(*midEQKnob_);

    highEQKnob_.reset(new juce::Slider("High EQ"));
    highEQKnob_->setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    highEQKnob_->setRange(-12.0, 12.0, 0.1);
    highEQKnob_->setValue(highEQ_.gain);
    highEQKnob_->onValueChange = [this] {
        setHighEQ(static_cast<float>(highEQKnob_->getValue()));
    };
    addAndMakeVisible(*highEQKnob_);

    // Compressor knobs (simplified)
    compThresholdKnob_.reset(new juce::Slider("Threshold"));
    compThresholdKnob_->setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    compThresholdKnob_->setRange(-60.0, 0.0, 0.1);
    compThresholdKnob_->setValue(-18.0);
    compThresholdKnob_->onValueChange = [this] {
        setCompressorThreshold(static_cast<float>(compThresholdKnob_->getValue()));
    };
    addAndMakeVisible(*compThresholdKnob_);

    compRatioKnob_.reset(new juce::Slider("Ratio"));
    compRatioKnob_->setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    compRatioKnob_->setRange(1.0, 20.0, 0.1);
    compRatioKnob_->setValue(4.0);
    compRatioKnob_->onValueChange = [this] {
        setCompressorRatio(static_cast<float>(compRatioKnob_->getValue()));
    };
    addAndMakeVisible(*compRatioKnob_);

    // Insert slots (ComboBox for selecting effects)
    for (int i = 0; i < insertSlots_.size(); ++i) {
        insertSlots_[i].reset(new juce::ComboBox(juce::String("Insert ") + juce::String(i + 1)));
        insertSlots_[i]->addItem("None", 1);
        insertSlots_[i]->addItem("Reverb", 2);
        insertSlots_[i]->addItem("Delay", 3);
        insertSlots_[i]->addItem("Chorus", 4);
        insertSlots_[i]->setSelectedId(1);
        addAndMakeVisible(*insertSlots_[i]);
    }

    // Meter display (placeholder)
    meterDisplay_.reset(new juce::Component("Meter"));
    addAndMakeVisible(*meterDisplay_);
}

void ChannelStrip::paint(juce::Graphics& g) {
    g.fillAll(getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId)); // Background
    g.setColour(juce::Colours::darkgrey);
    g.drawRect(getLocalBounds().toFloat(), 1.0f); // Border

    g.setColour(juce::Colours::white);
    g.setFont(14.0f);
    g.drawText(channelName_, getLocalBounds().reduced(5, 5).withHeight(20), juce::Justification::centredTop, true);

    // Draw meter
    juce::Rectangle<float> meterArea = getLocalBounds().reduced(5).withTop(getLocalBounds().getHeight() - 70).withHeight(60).toFloat();
    g.setColour(juce::Colours::green.withMultipliedAlpha(0.6f));
    g.fillRect(meterArea.withHeight(meterArea.getHeight() * (rmsLevel_ / 1.0f))); // Simple RMS meter
    g.setColour(juce::Colours::red.withMultipliedAlpha(0.8f));
    g.fillRect(meterArea.withHeight(meterArea.getHeight() * (peakLevel_ / 1.0f)).withY(meterArea.getY() + meterArea.getHeight() * (1.0f - peakLevel_ / 1.0f))); // Simple Peak meter
}

void ChannelStrip::resized() {
    juce::Rectangle<int> bounds = getLocalBounds();
    int padding = 5;
    int controlHeight = 25;
    int knobSize = 40;

    juce::Rectangle<int> currentBounds = bounds.reduced(padding);

    // Mute/Solo/Record buttons
    int buttonWidth = (currentBounds.getWidth() - padding * 2) / 3;
    muteButton_->setBounds(currentBounds.getX(), currentBounds.getY(), buttonWidth, controlHeight);
    soloButton_->setBounds(muteButton_->getRight() + padding, currentBounds.getY(), buttonWidth, controlHeight);
    recordArmButton_->setBounds(soloButton_->getRight() + padding, currentBounds.getY(), buttonWidth, controlHeight);

    currentBounds.removeFromTop(controlHeight + padding);

    // Pan knob
    panKnob_->setBounds(currentBounds.getX(), currentBounds.getY(), knobSize, knobSize);
    currentBounds.removeFromTop(knobSize + padding);

    // EQ knobs
    int eqKnobY = currentBounds.getY();
    lowEQKnob_->setBounds(currentBounds.getX(), eqKnobY, knobSize, knobSize);
    midEQKnob_->setBounds(lowEQKnob_->getRight() + padding, eqKnobY, knobSize, knobSize);
    highEQKnob_->setBounds(midEQKnob_->getRight() + padding, eqKnobY, knobSize, knobSize);
    currentBounds.removeFromTop(knobSize + padding);

    // Compressor knobs
    int compKnobY = currentBounds.getY();
    compThresholdKnob_->setBounds(currentBounds.getX(), compKnobY, knobSize, knobSize);
    compRatioKnob_->setBounds(compThresholdKnob_->getRight() + padding, compKnobY, knobSize, knobSize);
    currentBounds.removeFromTop(knobSize + padding);

    // Insert slots
    int insertSlotHeight = 20;
    for (int i = 0; i < insertSlots_.size(); ++i) {
        insertSlots_[i]->setBounds(currentBounds.getX(), currentBounds.getY() + i * (insertSlotHeight + padding), currentBounds.getWidth(), insertSlotHeight);
    }
    currentBounds.removeFromTop(insertSlots_.size() * (insertSlotHeight + padding));

    // Gain fader
    gainFader_->setBounds(currentBounds.getX(), currentBounds.getY(), currentBounds.getWidth(), currentBounds.getHeight() - (controlHeight + padding)); // Adjust for meter

    // Meter display
    meterDisplay_->setBounds(gainFader_->getX(), gainFader_->getBottom() + padding, gainFader_->getWidth(), controlHeight);
}

void ChannelStrip::setGain(float gainDb) { gainDb_ = gainDb; } // Placeholder
void ChannelStrip::setPan(float pan) { pan_ = pan; } // Placeholder
void ChannelStrip::setMute(bool muted) { muted_ = muted; } // Placeholder
void ChannelStrip::setSolo(bool soloed) { soloed_ = soloed; } // Placeholder
void ChannelStrip::setRecordArm(bool armed) { recordArmed_ = armed; } // Placeholder
void ChannelStrip::setLowEQ(float gain) { lowEQ_.gain = gain; } // Placeholder
void ChannelStrip::setMidEQ(float gain) { midEQ_.gain = gain; } // Placeholder
void ChannelStrip::setHighEQ(float gain) { highEQ_.gain = gain; } // Placeholder
void ChannelStrip::setLowFreq(float hz) { lowEQ_.frequency = hz; } // Placeholder
void ChannelStrip::setMidFreq(float hz) { midEQ_.frequency = hz; } // Placeholder
void ChannelStrip::setHighFreq(float hz) { highEQ_.frequency = hz; } // Placeholder
void ChannelStrip::setCompressorThreshold(float db) { /* ... */ } // Placeholder
void ChannelStrip::setCompressorRatio(float ratio) { /* ... */ } // Placeholder
void ChannelStrip::setCompressorAttack(float ms) { /* ... */ } // Placeholder
void ChannelStrip::setCompressorRelease(float ms) { /* ... */ } // Placeholder
void ChannelStrip::setCompressorMakeupGain(float db) { /* ... */ } // Placeholder
void ChannelStrip::setEQVisible(bool show) { /* ... */ } // Placeholder
void ChannelStrip::setCompressorVisible(bool show) { /* ... */ } // Placeholder
void ChannelStrip::setInsertVisible(bool show) { /* ... */ } // Placeholder
void ChannelStrip::setSendsVisible(bool show) { /* ... */ } // Placeholder
void ChannelStrip::setMetersVisible(bool show) { /* ... */ } // Placeholder
void ChannelStrip::addInsertEffect(InsertSlot slot, const std::string& effectName) { /* ... */ } // Placeholder
void ChannelStrip::removeInsertEffect(InsertSlot slot) { /* ... */ } // Placeholder
void ChannelStrip::bypassInsertEffect(InsertSlot slot, bool bypassed) { /* ... */ } // Placeholder
std::vector<std::string> ChannelStrip::getInsertEffectNames() const { return {}; } // Placeholder
void ChannelStrip::setSendLevel(int sendBus, float level) { /* ... */ } // Placeholder
void ChannelStrip::setSendPan(int sendBus, float pan) { /* ... */ } // Placeholder
void ChannelStrip::updateMeter(float peakLevel, float rmsLevel) { peakLevel_ = peakLevel; rmsLevel_ = rmsLevel; repaint(); } // Placeholder

//==============================================================================
// MixerConsolePanel Implementation
//==============================================================================

MixerConsolePanel::MixerConsolePanel()
    : viewMode_(ViewMode::MixerView),
      masterGainDb_(0.0f),
      showEQ_(true),
      showCompressor_(true),
      showInserts_(true),
      showSends_(true),
      showMeters_(true),
      theoryBrain_(std::make_unique<theory::MusicTheoryBrain>()) // Initialize MusicTheoryBrain
{
    // Initialize UI components for the mixer console
    channelContainer_.reset(new juce::Component("Channel Container"));
    channelViewport_.reset(new juce::Viewport("Channel Viewport"));
    channelViewport_->setViewedComponent(channelContainer_.get());
    addAndMakeVisible(*channelViewport_);

    masterChannel_.reset(new ChannelStrip("Master"));
    addAndMakeVisible(*masterChannel_);

    playButton_.reset(new juce::TextButton("Play"));
    addAndMakeVisible(*playButton_);
    stopButton_.reset(new juce::TextButton("Stop"));
    addAndMakeVisible(*stopButton_);
    recordButton_.reset(new juce::TextButton("Rec"));
    addAndMakeVisible(*recordButton_);

    viewModeSelector_.reset(new juce::ComboBox("View Mode"));
    viewModeSelector_->addItem("Mixer View", (int)ViewMode::MixerView + 1);
    viewModeSelector_->addItem("Track View", (int)ViewMode::TrackView + 1);
    viewModeSelector_->addItem("Compact View", (int)ViewMode::CompactView + 1);
    viewModeSelector_->addItem("Full View", (int)ViewMode::FullView + 1);
    viewModeSelector_->setSelectedId((int)viewMode_ + 1);
    viewModeSelector_->onChange = [this] {
        setViewMode((ViewMode)(viewModeSelector_->getSelectedId() - 1));
    };
    addAndMakeVisible(*viewModeSelector_);

    presetSelector_.reset(new juce::ComboBox("Mixer Preset"));
    presetSelector_->addItem("Default", 1);
    presetSelector_->onChange = [this] {
        onPresetSelected(presetSelector_->getSelectedId() - 1);
    };
    addAndMakeVisible(*presetSelector_);

    loadPresetButton_.reset(new juce::TextButton("Load"));
    addAndMakeVisible(*loadPresetButton_);
    savePresetButton_.reset(new juce::TextButton("Save"));
    addAndMakeVisible(*savePresetButton_);

    initializePresets();

    // Add some default channels for demonstration
    addChannel("Drums", "Drum Kit");
    addChannel("Bass", "Electric Bass");
    addChannel("Synth Lead", "Lead Synth");
}

// Destructor uses header default

void MixerConsolePanel::paint(juce::Graphics& g) {
    g.fillAll(juce::Colour(0xff3a3a3a)); // Dark background for mixer
}

void MixerConsolePanel::resized() {
    juce::Rectangle<int> bounds = getLocalBounds();

    int headerHeight = 40;
    int footerHeight = 40;
    int masterWidth = 80;

    // Layout header controls
    viewModeSelector_->setBounds(10, 10, 150, 24);
    presetSelector_->setBounds(bounds.getWidth() / 2 - 100, 10, 150, 24);
    loadPresetButton_->setBounds(presetSelector_->getRight() + 5, 10, 60, 24);
    savePresetButton_->setBounds(loadPresetButton_->getRight() + 5, 10, 60, 24);

    // Layout master channel
    masterChannel_->setBounds(bounds.getWidth() - masterWidth - 10, headerHeight, masterWidth, bounds.getHeight() - headerHeight - footerHeight);

    // Layout transport controls
    int transportButtonWidth = 60;
    int transportX = bounds.getWidth() / 2 - (transportButtonWidth * 3 + 10) / 2;
    playButton_->setBounds(transportX, bounds.getHeight() - footerHeight + 8, transportButtonWidth, 24);
    stopButton_->setBounds(playButton_->getRight() + 5, bounds.getHeight() - footerHeight + 8, transportButtonWidth, 24);
    recordButton_->setBounds(stopButton_->getRight() + 5, bounds.getHeight() - footerHeight + 8, transportButtonWidth, 24);

    // Layout channel viewport
    channelViewport_->setBounds(10, headerHeight, bounds.getWidth() - masterWidth - 30, bounds.getHeight() - headerHeight - footerHeight);

    // Adjust channel container size based on number of channels
    int totalChannelsWidth = 0;
    for (const auto& channel : channels_) {
        totalChannelsWidth += 100; // Assuming 100px per channel strip
    }
    channelContainer_->setSize(totalChannelsWidth, channelViewport_->getHeight());

    // Layout individual channel strips within the container
    int currentX = 0;
    for (auto& channel : channels_) {
        channel->setBounds(currentX, 0, 95, channelContainer_->getHeight()); // 95 width + 5 padding
        currentX += 100;
    }
}

int MixerConsolePanel::addChannel(const std::string& name, const std::string& instrument) {
    channels_.push_back(std::make_unique<ChannelStrip>(name));
    channelInstruments_.push_back(instrument);
    channelContainer_->addAndMakeVisible(*channels_.back());
    resized(); // Recalculate layout
    return (int)channels_.size() - 1;
}

void MixerConsolePanel::removeChannel(int channelIndex) {
    if (channelIndex >= 0 && channelIndex < channels_.size()) {
        channels_.erase(channels_.begin() + channelIndex);
        channelInstruments_.erase(channelInstruments_.begin() + channelIndex);
        resized();
    }
}

ChannelStrip* MixerConsolePanel::getChannel(int channelIndex) {
    if (channelIndex >= 0 && channelIndex < channels_.size()) {
        return channels_[channelIndex].get();
    }
    return nullptr;
}

std::vector<ChannelStrip*> MixerConsolePanel::getAllChannels() {
    std::vector<ChannelStrip*> allChannels;
    for (const auto& channel : channels_) {
        allChannels.push_back(channel.get());
    }
    return allChannels;
}

void MixerConsolePanel::loadPreset(const MixerPreset& preset) {
    // Clear existing channels
    channels_.clear();
    channelInstruments_.clear();

    // Load channels from preset
    for (const auto& channelSetup : preset.channels) {
        int index = addChannel(channelSetup.name, channelSetup.instrument);
        ChannelStrip* channel = getChannel(index);
        if (channel) {
            channel->setGain(channelSetup.gain);
            channel->setPan(channelSetup.pan);
            // Load insert effects etc.
        }
    }
    masterGainDb_ = preset.masterGain; // Now directly access masterGain
    setViewMode(ViewMode::MixerView); // Default to mixer view
    repaint();
}

void MixerConsolePanel::savePreset(const std::string& name) {
    MixerPreset newPreset;
    newPreset.name = name;
    newPreset.description = "Saved from current mixer state";
    newPreset.masterGain = masterGainDb_; // Directly set masterGain

    for (size_t i = 0; i < channels_.size(); ++i) {
        ChannelStrip* channel = channels_[i].get();
        if (channel) {
            newPreset.channels.push_back({
                channel->getName(), // .toStdString() is valid for juce::String
                channelInstruments_[i],
                channel->getGain(),
                channel->getPan(),
                channel->getInsertEffectNames() // Pass vector directly
            });
        }
    }
    presets_.push_back(newPreset);
}

std::vector<MixerConsolePanel::MixerPreset> MixerConsolePanel::getAvailablePresets() const {
    return presets_;
}

void MixerConsolePanel::loadRockBandTemplate() {
    MixerPreset preset;
    preset.name = "Rock Band";
    preset.description = "Standard rock band setup: Drums, Bass, Guitars, Vocals";
    preset.masterGain = 0.0f; // Directly set masterGain
    preset.channels.push_back({"Drums", "Drum Kit", 0.0f, 0.0f, {}});
    preset.channels.push_back({"Bass", "Electric Bass", 0.0f, 0.0f, {}});
    preset.channels.push_back({"Guitar R", "Electric Guitar", 0.0f, 0.8f, {}});
    preset.channels.push_back({"Guitar L", "Electric Guitar", 0.0f, -0.8f, {}});
    preset.channels.push_back({"Vocals", "Lead Vocal", 0.0f, 0.0f, {}});
    presets_.push_back(preset); // Add to internal presets storage
    loadPreset(preset);
}

void MixerConsolePanel::loadOrchestralTemplate() {
    MixerPreset preset;
    preset.name = "Orchestral";
    preset.description = "Classical orchestra setup: Strings, Brass, Woodwinds, Percussion";
    preset.masterGain = 0.0f;
    preset.channels.push_back({"Strings", "Orchestral Strings", 0.0f, 0.0f, {}});
    preset.channels.push_back({"Brass", "Orchestral Brass", 0.0f, 0.0f, {}});
    preset.channels.push_back({"Woodwinds", "Orchestral Woodwinds", 0.0f, 0.0f, {}});
    preset.channels.push_back({"Percussion", "Orchestral Percussion", 0.0f, 0.0f, {}});
    presets_.push_back(preset);
    loadPreset(preset);
}

void MixerConsolePanel::loadElectronicTemplate() {
    MixerPreset preset;
    preset.name = "Electronic";
    preset.description = "Electronic music setup: Synths, Drums, FX buses";
    preset.masterGain = 0.0f;
    preset.channels.push_back({"Kick", "Electronic Drums", 0.0f, 0.0f, {}});
    preset.channels.push_back({"Snare", "Electronic Drums", 0.0f, 0.0f, {}});
    preset.channels.push_back({"Hi-Hats", "Electronic Drums", 0.0f, 0.0f, {}});
    preset.channels.push_back({"Synth Pad", "Pad Synth", 0.0f, 0.0f, {}});
    preset.channels.push_back({"Synth Lead", "Lead Synth", 0.0f, 0.0f, {}});
    presets_.push_back(preset);
    loadPreset(preset);
}

void MixerConsolePanel::loadJazzComboTemplate() {
    MixerPreset preset;
    preset.name = "Jazz Combo";
    preset.description = "Jazz combo setup: Piano, Bass, Drums, Horns";
    preset.masterGain = 0.0f;
    preset.channels.push_back({"Piano", "Grand Piano", 0.0f, 0.0f, {}});
    preset.channels.push_back({"Bass", "Double Bass", 0.0f, 0.0f, {}});
    preset.channels.push_back({"Drums", "Jazz Drums", 0.0f, 0.0f, {}});
    preset.channels.push_back({"Saxophone", "Tenor Sax", 0.0f, 0.0f, {}});
    preset.channels.push_back({"Trumpet", "Trumpet", 0.0f, 0.0f, {}});
    presets_.push_back(preset);
    loadPreset(preset);
}

// Corrected definition for createSongwriterPreset
void MixerConsolePanel::createSongwriterPreset() {
    MixerPreset preset;
    preset.name = "Songwriter";
    preset.description = "Songwriter setup: Vocals, Acoustic Guitar, Piano";
    preset.masterGain = 0.0f;
    preset.channels.push_back({"Vocals", "Lead Vocal", 0.0f, 0.0f, {}});
    preset.channels.push_back({"Guitar", "Acoustic Guitar", 0.0f, 0.0f, {}});
    preset.channels.push_back({"Piano", "Grand Piano", 0.0f, 0.0f, {}});
    presets_.push_back(preset);
    loadPreset(preset);
}

void MixerConsolePanel::setMasterGain(float gainDb) {
    masterGainDb_ = gainDb;
    masterChannel_->setGain(gainDb);
}

void MixerConsolePanel::setMasterLimiter(bool enabled, float threshold) {
    // Placeholder for master limiter logic
    juce::ignoreUnused(enabled, threshold);
}

int MixerConsolePanel::addEffectBus(const std::string& name, const std::string& effectType) {
    effectBuses_.push_back(std::make_unique<ChannelStrip>(name + " (" + effectType + ")"));
    addAndMakeVisible(*effectBuses_.back());
    resized();
    return (int)effectBuses_.size() - 1;
}

void MixerConsolePanel::setEffectBusLevel(int busIndex, float level) {
    if (busIndex >= 0 && static_cast<size_t>(busIndex) < effectBuses_.size()) { // Corrected comparison
        effectBuses_[busIndex]->setGain(level); // Use gain for level
    }
}

void MixerConsolePanel::setEffectBusParameters(int busIndex, const std::string& paramName, float value) {
    // Placeholder for setting effect parameters
    juce::ignoreUnused(busIndex, paramName, value);
}

void MixerConsolePanel::routeChannelToOutput(int channelIndex, int outputBus) {
    // Placeholder for routing matrix logic
    juce::ignoreUnused(channelIndex, outputBus);
}

void MixerConsolePanel::createSubmix(const std::vector<int>& channelIndices, const std::string& submixName) {
    // Placeholder for submix creation logic
    juce::ignoreUnused(channelIndices, submixName);
}

void MixerConsolePanel::setChannelAutomationMode(int channelIndex, AutomationMode mode) {
    // Placeholder for automation mode
    juce::ignoreUnused(channelIndex, mode);
}

void MixerConsolePanel::recordAutomation(int channelIndex, const std::string& parameter, float value, double timestamp) {
    // Placeholder for recording automation
    juce::ignoreUnused(channelIndex, parameter, value, timestamp);
}

std::vector<MixerConsolePanel::AutomationPoint> MixerConsolePanel::getChannelAutomation(int channelIndex) const {
    // Placeholder for retrieving automation data
    juce::ignoreUnused(channelIndex);
    return {};
}

void MixerConsolePanel::setViewMode(ViewMode mode) {
    viewMode_ = mode;
    // Adjust layout based on view mode
    resized();
    repaint();
}

void MixerConsolePanel::setShowEQ(bool show) { showEQ_ = show; } // Placeholder
void MixerConsolePanel::setShowCompressor(bool show) { showCompressor_ = show; } // Placeholder
void MixerConsolePanel::setShowInserts(bool show) { showInserts_ = show; } // Placeholder
void MixerConsolePanel::setShowSends(bool show) { showSends_ = show; } // Placeholder
void MixerConsolePanel::setShowMeters(bool show) { showMeters_ = show; } // Placeholder

void MixerConsolePanel::routeMIDIToChannel(int channelIndex, const juce::MidiBuffer& midi) {
    if (channelIndex >= 0 && static_cast<size_t>(channelIndex) < channelMidi_.size()) { // Corrected comparison
        // The addEvents method needs a sampleDeltaToAdd parameter (typically 0)
        channelMidi_[channelIndex].addEvents(midi, 0, midi.getNumEvents(), 0);
    } else if (channelIndex == -1) {
        // Master channel or unassigned, merge with existing
        for (auto& entry : channelMidi_) {
            entry.second.addEvents(midi, 0, midi.getNumEvents(), 0);
        }
    }
}

juce::MidiBuffer MixerConsolePanel::getMixedOutput() const {
    juce::MidiBuffer mixedMidi;
    for (const auto& entry : channelMidi_) {
        mixedMidi.addEvents(entry.second, 0, entry.second.getNumEvents(), 0); // Merge all channel MIDI
    }
    return mixedMidi;
}

void MixerConsolePanel::applyMixToMIDI(juce::MidiBuffer& buffer, int channelIndex) {
    // Apply gain, pan, effects to MIDI notes (complex logic)
    juce::ignoreUnused(buffer, channelIndex);
}

void MixerConsolePanel::saveSnapshot(const std::string& name) {
    MixerSnapshot newSnapshot;
    newSnapshot.name = name;
    newSnapshot.timestamp = juce::Time::getCurrentTime().toMilliseconds();
    for (size_t i = 0; i < channels_.size(); ++i) {
        ChannelStrip* channel = channels_[i].get();
        if (channel) {
            newSnapshot.channelStates[static_cast<int>(i)] = {
                channel->getGain(),
                channel->getPan(),
                channel->isMuted(),
                channel->isSoloed(),
                {}
            };
        }
    }
    snapshots_.push_back(newSnapshot);
}

void MixerConsolePanel::loadSnapshot(const std::string& name) {
    for (const auto& snapshot : snapshots_) {
        if (snapshot.name == name) {
            for (const auto& entry : snapshot.channelStates) {
                int channelIndex = entry.first;
                const MixerSnapshot::ChannelState& state = entry.second;
                if (channelIndex >= 0 && static_cast<size_t>(channelIndex) < channels_.size()) { // Corrected comparison
                    ChannelStrip* channel = channels_[channelIndex].get();
                    if (channel) {
                        channel->setGain(state.gain);
                        channel->setPan(state.pan);
                        channel->setMute(state.muted);
                        channel->setSolo(state.soloed);
                        // Load other parameters
                    }
                }
            }
            // Assuming master channel is 0 or -1, use a safe check
            if (snapshot.channelStates.count(0)) { // Check for a master channel entry
                masterGainDb_ = snapshot.channelStates.at(0).gain; // Assuming master channel is index 0
            } else {
                masterGainDb_ = 0.0f; // Default if no master entry
            }
            repaint();
            return;
        }
    }
}

std::vector<MixerConsolePanel::MixerSnapshot> MixerConsolePanel::getSnapshots() const {
    return snapshots_;
}

bool MixerConsolePanel::exportSession(const juce::File& outputFile) {
    // Serialize mixer state to JSON or a custom format
    juce::ignoreUnused(outputFile);
    return false;
}

bool MixerConsolePanel::importSession(const juce::File& inputFile) {
    // Deserialize mixer state from file
    juce::ignoreUnused(inputFile);
    return false;
}

void MixerConsolePanel::initializePresets() {
    // Add default presets
    createRockBandPreset();
    createOrchestralPreset();
    createElectronicPreset();
    createJazzComboPreset();
    createSongwriterPreset();
}

void MixerConsolePanel::createRockBandPreset() {
    MixerPreset preset;
    preset.name = "Rock Band";
    preset.description = "Standard rock band setup: Drums, Bass, Guitars, Vocals";
    preset.masterGain = 0.0f;
    preset.channels.push_back({"Drums", "Drum Kit", 0.0f, 0.0f, {}});
    preset.channels.push_back({"Bass", "Electric Bass", 0.0f, 0.0f, {}});
    preset.channels.push_back({"Guitar R", "Electric Guitar", 0.0f, 0.8f, {}});
    preset.channels.push_back({"Guitar L", "Electric Guitar", 0.0f, -0.8f, {}});
    preset.channels.push_back({"Vocals", "Lead Vocal", 0.0f, 0.0f, {}});
    presets_.push_back(preset);
}

void MixerConsolePanel::createOrchestralPreset() {
    MixerPreset preset;
    preset.name = "Orchestral";
    preset.description = "Classical orchestra setup: Strings, Brass, Woodwinds, Percussion";
    preset.masterGain = 0.0f;
    preset.channels.push_back({"Strings", "Orchestral Strings", 0.0f, 0.0f, {}});
    preset.channels.push_back({"Brass", "Orchestral Brass", 0.0f, 0.0f, {}});
    preset.channels.push_back({"Woodwinds", "Orchestral Woodwinds", 0.0f, 0.0f, {}});
    preset.channels.push_back({"Percussion", "Orchestral Percussion", 0.0f, 0.0f, {}});
    presets_.push_back(preset);
}

void MixerConsolePanel::createElectronicPreset() {
    MixerPreset preset;
    preset.name = "Electronic";
    preset.description = "Electronic music setup: Synths, Drums, FX buses";
    preset.masterGain = 0.0f;
    preset.channels.push_back({"Kick", "Electronic Drums", 0.0f, 0.0f, {}});
    preset.channels.push_back({"Snare", "Electronic Drums", 0.0f, 0.0f, {}});
    preset.channels.push_back({"Hi-Hats", "Electronic Drums", 0.0f, 0.0f, {}});
    preset.channels.push_back({"Synth Pad", "Pad Synth", 0.0f, 0.0f, {}});
    preset.channels.push_back({"Synth Lead", "Lead Synth", 0.0f, 0.0f, {}});
    presets_.push_back(preset);
}

void MixerConsolePanel::createJazzComboPreset() {
    MixerPreset preset;
    preset.name = "Jazz Combo";
    preset.description = "Jazz combo setup: Piano, Bass, Drums, Horns";
    preset.masterGain = 0.0f;
    preset.channels.push_back({"Piano", "Grand Piano", 0.0f, 0.0f, {}});
    preset.channels.push_back({"Bass", "Double Bass", 0.0f, 0.0f, {}});
    preset.channels.push_back({"Drums", "Jazz Drums", 0.0f, 0.0f, {}});
    preset.channels.push_back({"Saxophone", "Tenor Sax", 0.0f, 0.0f, {}});
    preset.channels.push_back({"Trumpet", "Trumpet", 0.0f, 0.0f, {}});
    presets_.push_back(preset);
}

void MixerConsolePanel::layoutMixerView() {
    // Already handled in resized for now
}

void MixerConsolePanel::layoutTrackView() {
    // Placeholder for track view layout
}

void MixerConsolePanel::layoutCompactView() {
    // Placeholder for compact view layout
}

void MixerConsolePanel::onChannelGainChanged(int channelIndex, float gain) {
    // Handle gain change from a channel strip
    juce::ignoreUnused(channelIndex, gain);
}

void MixerConsolePanel::onChannelPanChanged(int channelIndex, float pan) {
    // Handle pan change from a channel strip
    juce::ignoreUnused(channelIndex, pan);
}

void MixerConsolePanel::onChannelMuteChanged(int channelIndex, bool muted) {
    // Handle mute change from a channel strip
    juce::ignoreUnused(channelIndex, muted);
}

void MixerConsolePanel::onChannelSoloChanged(int channelIndex, bool soloed) {
    // Handle solo change from a channel strip
    juce::ignoreUnused(channelIndex, soloed);
}

void MixerConsolePanel::onPresetSelected(int presetIndex) {
    if (presetIndex >= 0 && static_cast<size_t>(presetIndex) < presets_.size()) { // Corrected comparison
        loadPreset(presets_[presetIndex]);
    }
}

void MixerConsolePanel::onViewModeChanged() {
    setViewMode((ViewMode)(viewModeSelector_->getSelectedId() - 1));
}

} // namespace midikompanion
