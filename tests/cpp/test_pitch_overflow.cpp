#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <limits>

namespace penta {

struct Note {
    uint8_t pitch;
    uint8_t velocity;
    uint8_t channel;
    uint64_t timestamp;

    constexpr Note() : pitch(0), velocity(0), channel(0), timestamp(0) {}
    constexpr Note(uint8_t p, uint8_t v, uint8_t c = 0, uint64_t t = 0)
        : pitch(p), velocity(v), channel(c), timestamp(t) {}
};

struct Chord {
    std::array<bool, 12> pitchClass;
    uint8_t root;
    uint8_t quality;
    float confidence;

    Chord() : pitchClass{}, root(0), quality(0), confidence(0.0f) {}
};

} // namespace penta

namespace penta::harmony {

using penta::Note;
using penta::Chord;

class VoiceLeading {
public:
    struct Config {
        float maxVoiceDistance;
        float parallelPenalty;
        float contraryBonus;
        bool allowVoiceCrossing;

        Config()
            : maxVoiceDistance(12.0f)
            , parallelPenalty(5.0f)
            , contraryBonus(2.0f)
            , allowVoiceCrossing(false)
        {}
    };

    explicit VoiceLeading(const Config& config = Config{}) : config_(config) {}

    std::vector<Note> findOptimalVoicing(
        const Chord& targetChord,
        const std::vector<Note>& currentVoices,
        uint8_t targetOctave = 4
    ) const noexcept;

    void generateVoicingCandidates(
        const Chord& chord,
        uint8_t octave,
        std::vector<std::vector<Note>>& candidates
    ) const noexcept;

private:
    Config config_;
};

std::vector<Note> VoiceLeading::findOptimalVoicing(
    const Chord& targetChord,
    const std::vector<Note>& currentVoices,
    uint8_t targetOctave
) const noexcept {
    if (currentVoices.empty()) {
        std::vector<Note> result;
        for (int i = 0; i < 12; ++i) {
            if (targetChord.pitchClass[i]) {
                int p = static_cast<int>(targetOctave) * 12 + i;
                if (p > 127) continue;
                Note note;
                note.pitch = static_cast<uint8_t>(p);
                note.velocity = 80;
                note.timestamp = 0.0;
                result.push_back(note);
            }
        }
        return result;
    }
    return currentVoices;
}

void VoiceLeading::generateVoicingCandidates(
    const Chord& chord,
    uint8_t octave,
    std::vector<std::vector<Note>>& candidates
) const noexcept {
    std::vector<uint8_t> chordTones;
    for (int i = 0; i < 12; ++i) {
        if (chord.pitchClass[i]) {
            chordTones.push_back(i);
        }
    }
    if (chordTones.empty()) return;

    for (int oct = octave - 1; oct <= octave + 1; ++oct) {
        if (oct < 0 || oct > 8) continue;

        std::vector<Note> voices;
        for (uint8_t tone : chordTones) {
            int p = oct * 12 + tone;
            if (p > 127) continue;
            Note note;
            note.pitch = static_cast<uint8_t>(p);
            note.velocity = 80;
            note.timestamp = 0;
            voices.push_back(note);
        }
        std::sort(voices.begin(), voices.end(),
            [](const Note& a, const Note& b) { return a.pitch < b.pitch; });
        candidates.push_back(voices);
    }

    for (size_t bassIndex = 1; bassIndex < chordTones.size(); ++bassIndex) {
        std::vector<Note> voices;
        for (size_t i = 0; i < chordTones.size(); ++i) {
            size_t toneIndex = (bassIndex + i) % chordTones.size();
            uint8_t tone = chordTones[toneIndex];

            int p = static_cast<int>(octave) * 12 + tone;
            if (i > 0 && !voices.empty()
                && p <= static_cast<int>(voices.back().pitch)) {
                p += 12;
            }
            if (p > 127) continue;
            Note note;
            note.pitch = static_cast<uint8_t>(p);
            note.velocity = 80;
            note.timestamp = 0;
            voices.push_back(note);
        }
        candidates.push_back(voices);
    }
}

} // namespace penta::harmony

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        std::printf("  FAIL: %s\n", msg); \
        tests_failed++; \
    } else { \
        std::printf("  PASS: %s\n", msg); \
        tests_passed++; \
    } \
} while(0)

void test_normal_octave() {
    std::printf("--- test_normal_octave (octave 4, C major) ---\n");
    penta::harmony::VoiceLeading vl;
    penta::Chord chord;
    chord.pitchClass = {true, false, false, false, true, false, false, true,
                        false, false, false, false};
    chord.root = 0;

    auto result = vl.findOptimalVoicing(chord, {}, 4);
    TEST_ASSERT(result.size() == 3, "3 notes in C major triad");
    for (auto& n : result) {
        TEST_ASSERT(n.pitch <= 127, "pitch <= 127");
        TEST_ASSERT(n.pitch >= 48 && n.pitch <= 59,
                    "pitch in octave 4 range");
    }
    TEST_ASSERT(result[0].pitch == 48, "C4 = 48");
    TEST_ASSERT(result[1].pitch == 52, "E4 = 52");
    TEST_ASSERT(result[2].pitch == 55, "G4 = 55");
}

void test_high_octave_clamp() {
    std::printf("--- test_high_octave_clamp (octave 11, should clamp) ---\n");
    penta::harmony::VoiceLeading vl;
    penta::Chord chord;
    chord.pitchClass = {true, false, false, false, true, false, false, true,
                        false, false, false, false};
    chord.root = 0;

    auto result = vl.findOptimalVoicing(chord, {}, 11);
    // octave 11 => pitches 132, 136, 139 — all > 127
    TEST_ASSERT(result.empty(),
                "no notes produced when octave pushes all pitches > 127");
}

void test_octave_10_partial() {
    std::printf("--- test_octave_10_partial (octave 10, some pitches valid) ---\n");
    penta::harmony::VoiceLeading vl;
    penta::Chord chord;
    // C, E, G  => pitch classes 0, 4, 7
    chord.pitchClass = {true, false, false, false, true, false, false, true,
                        false, false, false, false};
    chord.root = 0;

    // octave 10 => C=120, E=124, G=127  (all valid)
    auto result = vl.findOptimalVoicing(chord, {}, 10);
    for (auto& n : result) {
        TEST_ASSERT(n.pitch <= 127, "pitch <= 127");
    }
    TEST_ASSERT(result.size() == 3, "all 3 tones fit at octave 10");
    TEST_ASSERT(result[0].pitch == 120, "C10 = 120");
    TEST_ASSERT(result[1].pitch == 124, "E10 = 124");
    TEST_ASSERT(result[2].pitch == 127, "G10 = 127");
}

void test_generate_candidates_high_octave() {
    std::printf("--- test_generate_candidates_high_octave (octave 10) ---\n");
    penta::harmony::VoiceLeading vl;
    penta::Chord chord;
    // B major triad: B(11), D#(3), F#(6)
    chord.pitchClass = {false, false, false, true, false, false, true, false,
                        false, false, false, true};
    chord.root = 11;

    std::vector<std::vector<penta::Note>> candidates;
    vl.generateVoicingCandidates(chord, 10, candidates);

    for (size_t c = 0; c < candidates.size(); ++c) {
        for (auto& n : candidates[c]) {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                "candidate[%zu] pitch %d <= 127", c, (int)n.pitch);
            TEST_ASSERT(n.pitch <= 127, buf);
        }
    }
}

void test_inversion_overflow_guard() {
    std::printf("--- test_inversion_overflow_guard (octave 10 inversions) ---\n");
    penta::harmony::VoiceLeading vl;
    penta::Chord chord;
    // C major: C(0), E(4), G(7)
    chord.pitchClass = {true, false, false, false, true, false, false, true,
                        false, false, false, false};
    chord.root = 0;

    std::vector<std::vector<penta::Note>> candidates;
    vl.generateVoicingCandidates(chord, 10, candidates);

    for (size_t c = 0; c < candidates.size(); ++c) {
        for (auto& n : candidates[c]) {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                "inversion candidate[%zu] pitch %d <= 127", c, (int)n.pitch);
            TEST_ASSERT(n.pitch <= 127, buf);
        }
    }
}

int main() {
    std::printf("=== uint8_t pitch overflow tests ===\n\n");

    test_normal_octave();
    test_high_octave_clamp();
    test_octave_10_partial();
    test_generate_candidates_high_octave();
    test_inversion_overflow_guard();

    std::printf("\n=== Results: %d passed, %d failed ===\n",
                tests_passed, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
