#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include "penta/harmony/HarmonyEngine.h"
#include "penta/harmony/ChordAnalyzer.h"
#include "penta/common/RTTypes.h"

TEST_CASE("HarmonyEngine latency benchmark", "[harmony][performance][benchmark]") {
    penta::harmony::HarmonyEngine::Config config;
    config.sampleRate = penta::kDefaultSampleRate;
    config.analysisWindowSize = 2048;
    penta::harmony::HarmonyEngine harmonyEngine(config);

    const std::array<penta::Note, 3> notes = {
        penta::Note{60, 100},
        penta::Note{64, 100},
        penta::Note{67, 100},
    };

    BENCHMARK("HarmonyEngine processNotes") {
        harmonyEngine.processNotes(notes.data(), notes.size());
        return harmonyEngine.getCurrentChord().confidence;
    };

    const auto& chord = harmonyEngine.getCurrentChord();
    INFO("Chord root: " << static_cast<int>(chord.root));
}

TEST_CASE("ChordAnalyzer latency benchmark", "[harmony][performance][benchmark]") {
    penta::harmony::ChordAnalyzer analyzer;
    std::array<bool, 12> pitchClassSet{};
    pitchClassSet[0] = true;
    pitchClassSet[4] = true;
    pitchClassSet[7] = true;

    BENCHMARK("ChordAnalyzer analyze") {
        return analyzer.analyze(pitchClassSet);
    };

    // The SIMD version is automatically used if AVX2 is available, so this benchmark will reflect that.
    // No specific assertion here, just to get timing.
}
