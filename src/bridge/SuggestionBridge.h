#pragma once
/*
 * SuggestionBridge.h - Python Suggestion Engine Bridge
 * ====================================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - Python Layer: SuggestionEngine (intelligent suggestion generation)
 * - UI Layer: Used by EmotionWorkstation (SuggestionOverlay)
 * - Engine Layer: Provides suggestions based on current musical state
 * - Used By: EmotionWorkstation, MidiGenerator (for intelligent suggestions)
 *
 * Purpose: C++ interface to Python SuggestionEngine for intelligent suggestions
 *          without requiring direct Python embedding in audio thread.
 */

#include "bridge/PythonBridgeBase.h"
#include <string>
#include <vector>
#include <memory>

// Forward declaration
struct _object;
typedef struct _object PyObject;

namespace kelly {

/**
 * SuggestionBridge - C++ interface to Python SuggestionEngine
 *
 * Provides methods to get intelligent suggestions from Python SuggestionEngine
 * without requiring direct Python embedding in audio thread.
 */
class SuggestionBridge : public bridge::PythonBridgeBase {
public:
    SuggestionBridge();
    ~SuggestionBridge() override;

    // BridgeBase interface
    bool initialize() override;
    void shutdown() override;

    /**
     * Get suggestions based on current musical state.
     *
     * @param currentStateJson JSON string with current state:
     *   {
     *     "parameters": {"valence": -0.5, "arousal": 0.4, ...},
     *     "emotion": "grief",
     *     "rule_breaks": ["HARMONY_AvoidTonicResolution"]
     *   }
     * @param maxSuggestions Maximum number of suggestions to return
     * @return JSON string with suggestions array
     */
    std::string getSuggestions(
        const std::string& currentStateJson,
        int maxSuggestions = 5
    );

    /**
     * Record that a suggestion was shown to the user.
     */
    void recordSuggestionShown(
        const std::string& suggestionId,
        const std::string& suggestionType,
        const std::string& contextJson = "{}"
    );

    /**
     * Record that user accepted (applied) a suggestion.
     */
    void recordSuggestionAccepted(const std::string& suggestionId);

    /**
     * Record that user dismissed a suggestion.
     */
    void recordSuggestionDismissed(const std::string& suggestionId);

private:
    // Python function pointers
    PyObject* getSuggestionsFunc_ = nullptr;
    PyObject* recordShownFunc_ = nullptr;
    PyObject* recordAcceptedFunc_ = nullptr;
    PyObject* recordDismissedFunc_ = nullptr;
    PyObject* module_ = nullptr;
};

} // namespace kelly
