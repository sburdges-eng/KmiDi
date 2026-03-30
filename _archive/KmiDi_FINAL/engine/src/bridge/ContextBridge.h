#pragma once
/*
 * ContextBridge.h - Python Context Analyzer Bridge
 * =================================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - Python Layer: ContextAnalyzer (musical context analysis)
 * - MIDI Layer: Used by MidiGenerator and engines (context-aware generation)
 * - Engine Layer: Provides context analysis for generation decisions
 * - Used By: MidiGenerator, various engines (for context-aware parameters)
 *
 * Purpose: C++ interface to Python ContextAnalyzer for analyzing musical context
 *          and getting context-aware parameters for C++ engines.
 *
 * Thread Safety:
 * - All methods are thread-safe
 * - Python calls are cached to avoid blocking
 * - Context is updated asynchronously
 */

#include "bridge/PythonBridgeBase.h"
#include "bridge/CacheManager.h"
#include <string>
#include <map>
#include <memory>

// Forward declaration
struct _object;
typedef struct _object PyObject;

namespace kelly {

/**
 * ContextBridge - C++ interface to Python ContextAnalyzer
 *
 * Provides methods to analyze musical context and get context-aware parameters
 * for C++ engines. Context analysis informs generation decisions.
 *
 * Thread Safety:
 * - All methods are thread-safe
 * - Python calls are cached to avoid blocking
 * - Context is updated asynchronously
 */
class ContextBridge : public bridge::PythonBridgeBase {
public:
    ContextBridge();
    ~ContextBridge() override;

    // BridgeBase interface
    bool initialize() override;
    void shutdown() override;

    /**
     * Analyze current musical context.
     *
     * @param stateJson JSON string with current state:
     *   {
     *     "emotion": "grief",
     *     "parameters": {"valence": -0.5, "arousal": 0.4, ...},
     *     "chords": ["Am", "Dm", "F", "C"],
     *     "current_section": "verse",
     *     ...
     *   }
     * @return JSON string with context analysis:
     *   {
     *     "emotion_category": "negative_low_energy",
     *     "complexity_level": "low",
     *     "parameter_ranges": {"valence": "low", "arousal": "medium", ...},
     *     "harmonic_state": "tonic",
     *     "rhythmic_state": "straight",
     *     "suggestions": ["Consider slower tempo", ...]
     *   }
     */
    std::string analyzeContext(const std::string& stateJson);

    /**
     * Get context-aware parameter adjustments.
     *
     * @param stateJson Current state JSON
     * @return JSON string with parameter adjustments:
     *   {
     *     "tempo": 70,
     *     "complexity": 0.3,
     *     "density": 0.2,
     *     "justification": "Low energy emotions benefit from slower tempos"
     *   }
     */
    std::string getContextualParameters(const std::string& stateJson);

    /**
     * Update context with new state information.
     *
     * @param stateJson Updated state JSON
     */
    void updateContext(const std::string& stateJson);

    /**
     * Get contextual suggestions based on analysis.
     *
     * @param stateJson Current state JSON
     * @return JSON string with suggestions:
     *   {
     *     "suggestions": [
     *       "Consider increasing complexity for more interest",
     *       "Low energy emotions often benefit from slower tempos"
     *     ]
     *   }
     */
    std::string getContextualSuggestions(const std::string& stateJson);

    /**
     * Clear context cache (force fresh analysis on next call).
     */
    void clearCache();

private:
    // Python function pointers
    PyObject* analyzeContextFunc_ = nullptr;
    PyObject* getContextualParametersFunc_ = nullptr;
    PyObject* updateContextFunc_ = nullptr;
    PyObject* getSuggestionsFunc_ = nullptr;
    PyObject* module_ = nullptr;

    // Context cache
    bridge::CacheManager cache_;
    static constexpr int CACHE_TTL_MS = 2000;  // Cache for 2 seconds

    std::string hashState(const std::string& stateJson);
};

} // namespace kelly
