#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <string>
#include <vector>

// Teaching mode overlay component that explains musical concepts.
// Read-only annotations based on Intent Schema and rule-breaking database.
class TeachingOverlayComponent : public juce::Component {
public:
    TeachingOverlayComponent();
    ~TeachingOverlayComponent() override = default;

    void paint(juce::Graphics& g) override;
    void resized() override;

    // Set teaching mode enabled/disabled
    void setEnabled(bool enabled);
    bool isEnabled() const { return enabled_; }

    // Load intent schema JSON path (read-only)
    void setIntentSchemaPath(const std::string& path);

    // Update annotations based on current timeline position
    void updateAnnotations(float timelinePosition, float timelineWidth);

private:
    struct Annotation {
        std::string text;
        juce::Rectangle<float> bounds;
        float priority; // Higher = more important
    };

    void loadIntentSchema();
    void generateAnnotations(float timelinePosition, float timelineWidth);
    juce::Rectangle<float> getAnnotationBounds(const Annotation& ann, float timelineWidth) const;

    bool enabled_ {false};
    std::string intentSchemaPath_;
    std::vector<Annotation> annotations_;

    // Cached layout
    juce::Font annotationFont_;
    float lastTimelinePosition_ {0.0f};
    float lastTimelineWidth_ {0.0f};
};
