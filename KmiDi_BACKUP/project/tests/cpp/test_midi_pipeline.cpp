#include <catch2/catch_test_macros.hpp>
#include "core/midi_pipeline.h"

using namespace kelly;

TEST_CASE("MidiPipeline initializes with default tempo", "[midi]") {
    MidiPipeline pipeline;
    REQUIRE(pipeline.getTempo() == 120);
}

TEST_CASE("MidiPipeline can set tempo", "[midi]") {
    MidiPipeline pipeline;
    pipeline.setTempo(140);
    REQUIRE(pipeline.getTempo() == 140);
}

TEST_CASE("MidiPipeline can add notes", "[midi]") {
    MidiPipeline pipeline;
    MidiNote note{60, 100, 0, 480};
    pipeline.addNote(note);
    REQUIRE(pipeline.getNotes().size() == 1);
}

TEST_CASE("MidiPipeline can clear notes", "[midi]") {
    MidiPipeline pipeline;
    MidiNote note{60, 100, 0, 480};
    pipeline.addNote(note);
    pipeline.clear();
    REQUIRE(pipeline.getNotes().empty());
}
