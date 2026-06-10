#include "penta/harmony/ChordAnalyzer.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

// The "SIMD" path in this file is really bitmask+popcount, not vector math —
// on any recent x86-64 (POPCNT, ≥SSE4.2) or ARMv8, __builtin_popcount
// compiles to a single instruction (POPCNT or CNT). No ISA intrinsics are
// required, so we drop the AVX2 gate entirely and provide one portable
// implementation for everyone. On Apple Silicon this was previously a
// scalar-fallback (#ifndef __AVX2__ branch in this file) that also
// duplicated ChordAnalyzer.cpp's own scalar symbols — silent ODR violation
// on arm64. That is fixed here: this TU now has exactly one definition of
// each *SIMD function, valid on every supported ISA. ChordAnalyzer.cpp's
// scalar fallback is removed in the same commit to keep ownership single.
//
// The previous code also had an unused `__m256 vBestScore` + `_mm256_max_ps`
// pair whose result was never read — dead SIMD that the scalar tracking
// loop below already subsumed. Removed.

namespace penta::harmony {

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace {
// Single-instruction popcount on both x86-64 (POPCNT) and ARMv8 (CNT).
// std::popcount would be ideal but not all TUs build with C++20 yet.
// __builtin_popcount is gcc/clang-only; MSVC spells it __popcnt.
inline int popcount16(std::uint32_t x) noexcept {
#if defined(_MSC_VER)
  return static_cast<int>(__popcnt(x));
#else
  return __builtin_popcount(x);
#endif
}
} // namespace

float ChordAnalyzer::scoreAgainstTemplateSIMD(
    const std::array<bool, 12> &pitchClassSet, const ChordTemplate &template_,
    uint8_t root) const noexcept {
  // Pack bool arrays into 12-bit masks so match/extra/required are
  // single bitwise ops + popcount instead of 12 branches per call.
  std::uint32_t templateMask = 0;
  std::uint32_t inputMask = 0;

  for (int i = 0; i < 12; ++i) {
    if (template_.pattern[i]) {
      templateMask |= (1u << i);
    }
    int rotated = (i + root) % 12;
    if (pitchClassSet[rotated]) {
      inputMask |= (1u << i);
    }
  }

  const std::uint32_t matches = templateMask & inputMask;
  const std::uint32_t extra = inputMask & ~templateMask;
  const std::uint32_t required = templateMask;

  const int matchCount = popcount16(matches);
  const int requiredCount = popcount16(required);
  const int extraCount = popcount16(extra);

  if (requiredCount == 0)
    return 0.0f;

  const float completeness = static_cast<float>(matchCount) / requiredCount;
  const float extraPenalty = 1.0f / (1.0f + 0.5f * extraCount);

  return completeness * extraPenalty;
}

void ChordAnalyzer::findBestMatchSIMD(const std::array<bool, 12> &pitchClassSet,
                                      Chord &outChord) noexcept {
  // Pack input once; rotate via shift/OR rather than re-reading the
  // bool array for every root.
  std::uint32_t inputMask = 0;
  int pitchCount = 0;
  for (int i = 0; i < 12; ++i) {
    if (pitchClassSet[i]) {
      inputMask |= (1u << i);
      ++pitchCount;
    }
  }

  if (pitchCount == 0) {
    outChord.confidence = 0.0f;
    return;
  }

  float bestScore = 0.0f;
  std::uint8_t bestRoot = 0;
  std::uint8_t bestQuality = 0;

  for (std::uint8_t root = 0; root < 12; ++root) {
    // Rotate inputMask left by `root` within the 12-bit ring.
    const std::uint32_t rotatedInput =
        ((inputMask << root) | (inputMask >> (12 - root))) & 0xFFFu;

    for (const auto &tmpl : kChordTemplates) {
      std::uint32_t templateMask = 0;
      for (int j = 0; j < 12; ++j) {
        if (tmpl.pattern[j]) {
          templateMask |= (1u << j);
        }
      }

      const std::uint32_t matches = rotatedInput & templateMask;
      const std::uint32_t extra = rotatedInput & ~templateMask;

      const int matchCount = popcount16(matches);
      const int requiredCount = popcount16(templateMask);
      const int extraCount = popcount16(extra);

      if (requiredCount == 0) {
        continue;
      }

      const float completeness = static_cast<float>(matchCount) / requiredCount;
      const float extraPenalty = 1.0f / (1.0f + 0.5f * extraCount);
      const float score = completeness * extraPenalty;

      if (score > bestScore) {
        bestScore = score;
        bestRoot = root;
        bestQuality = tmpl.quality;
      }
    }
  }

  outChord.root = bestRoot;
  outChord.quality = bestQuality;
  outChord.confidence = bestScore;
  outChord.pitchClass = pitchClassSet;
}

Chord ChordAnalyzer::analyzeSIMD(
    const std::array<bool, 12> &pitchClassSet) noexcept {
  Chord result;
  findBestMatchSIMD(pitchClassSet, result);
  return result;
}

} // namespace penta::harmony
