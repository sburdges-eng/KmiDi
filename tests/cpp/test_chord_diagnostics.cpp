#include <catch2/catch_test_macros.hpp>
#include "core/chord_diagnostics.h"

using namespace kelly;

TEST_CASE("ChordDiagnostics calculates dissonance", "[chord]") {
    ChordDiagnostics diagnostics;
    Chord chord{{60, 64, 67}, "C major"};
    float dissonance = diagnostics.calculateDissonance(chord);
    REQUIRE(dissonance >= 0.0f);
}

TEST_CASE("ChordDiagnostics identifies major chord", "[chord]") {
    ChordDiagnostics diagnostics;
    std::vector<uint8_t> notes{60, 64, 67};  // C major
    std::string type = diagnostics.identifyChord(notes);
    REQUIRE(type == "major");
}

TEST_CASE("ChordDiagnostics identifies minor chord", "[chord]") {
    ChordDiagnostics diagnostics;
    std::vector<uint8_t> notes{60, 63, 67};  // C minor
    std::string type = diagnostics.identifyChord(notes);
    REQUIRE(type == "minor");
}

TEST_CASE("ChordDiagnostics checks consonance", "[chord]") {
    ChordDiagnostics diagnostics;
    Chord chord{{60, 64, 67}, "C major"};
    bool consonant = diagnostics.isConsonant(chord, 0.5f);
    REQUIRE(consonant == true);
}
