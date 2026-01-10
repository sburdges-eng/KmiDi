#pragma once
/*
 * EngineIntelligenceBridge.h - Python Engine Intelligence Bridge
 * ===============================================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - Python Layer: Python engine-level intelligence modules
 * - Engine Layer: Used by MelodyEngine, BassEngine, DrumGrooveEngine, etc.
 * - MIDI Layer: Engines use this bridge for intelligent suggestions
 * - Used By: All MIDI engines (MelodyEngine, BassEngine, PadEngine, etc.)
 *
 * Purpose: C++ interface to Python engine-level intelligence for getting intelligent
 *          suggestions for individual C++ engines from Python intelligence modules.
 *
 * Thread Safety:
 * - All methods are thread-safe
 * - Python calls are cached to avoid blocking audio thread
 * - Suggestions are returned as JSON strings for easy parsing
 */

#include "bridge/PythonBridgeBase.h"
#include "bridge/CacheManager.h"
#include <string>
#include <vector>
#include <map>
#include <memory>

// Forward declaration
struct _object;
typedef struct _object PyObject;

namespace kelly {

/**
 * EngineIntelligenceBridge - C++ interface to Python engine-level intelligence
 *
 * Provides methods to get intelligent suggestions for individual C++ engines
 * (MelodyEngine, BassEngine, DrumGrooveEngine, etc.) from Python intelligence modules.
 *
 * Thread Safety:
 * - All methods are thread-safe
 * - Python calls are cached to avoid blocking audio thread
 * - Suggestions are returned as JSON strings for easy parsing
 */
class EngineIntelligenceBridge : public bridge::PythonBridgeBase {
public:
    EngineIntelligenceBridge();
    ~EngineIntelligenceBridge() override;

    // BridgeBase interface
    bool initialize() override;
    void shutdown() override;

    /**
     * Get suggestions for a specific engine type.
     *
     * @param engineType Engine type: "melody", "bass", "drum", "pad", "string", etc.
     * @param currentStateJson JSON string with current state:
     *   {
     *     "emotion": "grief",
     *     "key": "C",
     *     "mode": "minor",
     *     "chords": ["Am", "Dm", "F", "C"],
     *     "parameters": {"complexity": 0.4, "density": 0.3, ...},
     *     "context": {"emotion_category": "negative_low_energy", ...}
     *   }
     * @return JSON string with engine-specific suggestions:
     *   {
     *     "contour": "descending",
     *     "density": "sparse",
     *     "velocity_range": [40, 75],
     *     "register_range": [55, 75],
     *     "parameter_adjustments": {"complexity": 0.3, "density": 0.2},
     *     "confidence": 0.85
     *   }
     */
    std::string getEngineSuggestions(
        const std::string& engineType,
        const std::string& currentStateJson
    );

    /**
     * Get suggestions for multiple engines at once (batch operation).
     *
     * @param engineTypes Vector of engine types
     * @param currentStateJson Current state JSON
     * @return JSON string with suggestions for all engines:
     *   {
     *     "melody": {...},
     *     "bass": {...},
     *     "drum": {...}
     *   }
     */
    std::string getBatchEngineSuggestions(
        const std::vector<std::string>& engineTypes,
        const std::string& currentStateJson
    );

    /**
     * Record that engine suggestions were applied.
     *
     * @param engineType Engine type
     * @param suggestionJson The suggestion that was applied
     * @param resultJson Result of applying the suggestion (for learning)
     */
    void recordSuggestionApplied(
        const std::string& engineType,
        const std::string& suggestionJson,
        const std::string& resultJson = "{}"
    );

    /**
     * Clear suggestion cache (force fresh suggestions on next call).
     */
    void clearCache();

    /**
     * Report engine state to Python (for learning/feedback).
     */
    void reportEngineState(
        const std::string& engineType,
        const std::string& stateJson,
        const std::string& generatedNotesJson
    );

private:
    // Python function pointers
    PyObject* getEngineSuggestionsFunc_ = nullptr;
    PyObject* getBatchSuggestionsFunc_ = nullptr;
    PyObject* recordAppliedFunc_ = nullptr;
    PyObject* reportEngineStateFunc_ = nullptr;
    PyObject* module_ = nullptr;

    // Suggestion cache
    bridge::CacheManager cache_;
    static constexpr int CACHE_TTL_MS = 1000;  // Cache for 1 second

    std::string hashState(const std::string& stateJson);
};

} // namespace kelly
