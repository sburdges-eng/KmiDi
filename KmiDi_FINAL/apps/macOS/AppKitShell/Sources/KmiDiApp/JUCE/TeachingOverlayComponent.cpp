#include "TeachingOverlayComponent.h"
#include <juce_graphics/juce_graphics.h>
#include <fstream>
#include <sstream>
#include <algorithm>

TeachingOverlayComponent::TeachingOverlayComponent() {
    setInterceptsMouseClicks(false, false); // Pass through mouse events
    annotationFont_ = juce::Font(11.0f);
}

void TeachingOverlayComponent::setEnabled(bool enabled) {
    if (enabled_ == enabled)
        return;
    enabled_ = enabled;
    setVisible(enabled);
    if (enabled) {
        loadIntentSchema();
    } else {
        annotations_.clear();
    }
    repaint();
}

void TeachingOverlayComponent::setIntentSchemaPath(const std::string& path) {
    intentSchemaPath_ = path;
    if (enabled_)
        loadIntentSchema();
}

void TeachingOverlayComponent::paint(juce::Graphics& g) {
    if (!enabled_ || annotations_.empty())
        return;

    g.setFont(annotationFont_);

    for (const auto& ann : annotations_) {
        const auto bounds = getAnnotationBounds(ann, (float)getWidth());

        // Background for annotation
        g.setColour(juce::Colour(0x80000000)); // Semi-transparent black
        g.fillRoundedRectangle(bounds.expanded(4.0f), 3.0f);

        // Text
        g.setColour(juce::Colours::white.withAlpha(0.9f));
        g.drawText(ann.text, bounds, juce::Justification::centredLeft, true);
    }
}

void TeachingOverlayComponent::resized() {
    // Recalculate annotations if timeline dimensions changed
    if (enabled_ && (std::abs(lastTimelineWidth_ - (float)getWidth()) > 1.0f)) {
        updateAnnotations(lastTimelinePosition_, (float)getWidth());
    }
}

void TeachingOverlayComponent::updateAnnotations(float timelinePosition, float timelineWidth) {
    if (!enabled_)
        return;

    lastTimelinePosition_ = timelinePosition;
    lastTimelineWidth_ = timelineWidth;

    generateAnnotations(timelinePosition, timelineWidth);
    repaint();
}

void TeachingOverlayComponent::loadIntentSchema() {
    annotations_.clear();

    if (intentSchemaPath_.empty())
        return;

    // Simple JSON parsing: extract rule_to_break and justification
    std::ifstream file(intentSchemaPath_);
    if (!file.is_open())
        return;

    std::string line;
    std::string ruleToBreak;
    std::string justification;

    while (std::getline(file, line)) {
        // Look for rule_to_break
        const auto rulePos = line.find("\"technical_rule_to_break\":");
        if (rulePos != std::string::npos) {
            const auto start = line.find("\"", rulePos + 27) + 1;
            const auto end = line.find("\"", start);
            if (end != std::string::npos) {
                ruleToBreak = line.substr(start, end - start);
            }
        }

        // Look for justification
        const auto justPos = line.find("\"rule_breaking_justification\":");
        if (justPos != std::string::npos) {
            const auto start = line.find("\"", justPos + 31) + 1;
            const auto end = line.find("\"", start);
            if (end != std::string::npos) {
                justification = line.substr(start, end - start);
            }
        }
    }

    // Generate annotation from rule-breaking logic
    if (!ruleToBreak.empty() && !justification.empty()) {
        Annotation ann;
        ann.text = ruleToBreak + ": " + justification;
        ann.priority = 1.0f;
        annotations_.push_back(ann);
    }
}

void TeachingOverlayComponent::generateAnnotations(float timelinePosition, float timelineWidth) {
    // For now, show annotations at fixed positions
    // In a full implementation, this would map timeline position to musical events
    // and generate contextual annotations

    if (annotations_.empty())
        return;

    // Position annotations near the playhead
    for (auto& ann : annotations_) {
        ann.bounds = juce::Rectangle<float>(
            timelinePosition + 20.0f,
            20.0f,
            200.0f,
            30.0f
        );
    }
}

juce::Rectangle<float> TeachingOverlayComponent::getAnnotationBounds(
    const Annotation& ann,
    float timelineWidth) const {
    // Ensure annotation stays within bounds
    auto bounds = ann.bounds;
    if (bounds.getRight() > timelineWidth) {
        bounds.setX(timelineWidth - bounds.getWidth() - 10.0f);
    }
    if (bounds.getX() < 0.0f) {
        bounds.setX(10.0f);
    }
    return bounds;
}
