#pragma once
/*
 * WorkstationPanel.h - Multi-Track MIDI Workstation UI
 * =====================================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - UI Layer: EmotionWorkstation / PluginEditor (hosts panel)
 * - Common Layer: Types.h (EmotionNode, track metadata)
 * - MIDI Layer: Generated MIDI buffers / editors (visualization)
 *
 * Purpose: DAW-like track list with mute/solo/volume for generated parts.
 *
 * Features:
 * - Per-track controls and metering hooks
 * - Timer-driven refresh for live updates
 */

#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include "../common/Types.h"
#include <vector>
#include <memory>

namespace kelly {

class WorkstationPanel : public juce::Component,
                         public juce::Timer {
public:
    struct TrackInfo {
        std::string name;
        std::string icon;  // Unicode icon or emoji
        int channel = 0;
        int instrument = 0;
        bool muted = false;
        bool soloed = false;
        float volume = 1.0f;
        float pan = 0.0f;
        std::vector<MidiNote> notes;
        juce::Colour trackColour;
    };
    
    WorkstationPanel();
    ~WorkstationPanel() override;
    
    void paint(juce::Graphics& g) override;
    void resized() override;
    void timerCallback() override;
    
    // Update with generated MIDI data
    void updateTracks(const GeneratedMidi& midi);
    
    // Get track settings
    std::vector<TrackInfo> getTrackInfos() const { return tracks_; }
    
    // Callbacks
    std::function<void(int trackIndex, bool muted)> onTrackMuteChanged;
    std::function<void(int trackIndex, bool soloed)> onTrackSoloChanged;
    std::function<void(int trackIndex, float volume)> onTrackVolumeChanged;
    std::function<void(int trackIndex, float pan)> onTrackPanChanged;
    std::function<void(int trackIndex)> onTrackSelected;
    
private:
    static constexpr int TRACK_HEIGHT = 60;
    static constexpr int TRACK_HEADER_WIDTH = 200;
    static constexpr int CONTROL_WIDTH = 80;
    
    std::vector<TrackInfo> tracks_;
    int selectedTrackIndex_ = -1;
    int hoveredTrackIndex_ = -1;
    
    // Track controls
    struct TrackControl {
        juce::ToggleButton muteButton;
        juce::ToggleButton soloButton;
        juce::Slider volumeSlider;
        juce::Slider panSlider;
        juce::Label nameLabel;
        juce::Label instrumentLabel;
    };
    
    std::vector<std::unique_ptr<TrackControl>> trackControls_;
    
    // Scroll view
    juce::Viewport viewport_;
    juce::Component trackList_;
    
    // Track visualization
    void paintTrack(juce::Graphics& g, const TrackInfo& track, int index, const juce::Rectangle<int>& bounds);
    void paintTrackHeader(juce::Graphics& g, const TrackInfo& track, int index, const juce::Rectangle<int>& bounds);
    void paintTrackContent(juce::Graphics& g, const TrackInfo& track, const juce::Rectangle<int>& bounds);
    
    // Mouse interaction
    void mouseDown(const juce::MouseEvent& e) override;
    void mouseMove(const juce::MouseEvent& e) override;
    void mouseExit(const juce::MouseEvent& e) override;
    
    // Initialize default tracks
    void initializeTracks();
    
    // Update track controls visibility
    void updateTrackControls();
    
    // Get track index from mouse position
    int getTrackIndexFromPosition(const juce::Point<int>& pos) const;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(WorkstationPanel)
};

} // namespace kelly
