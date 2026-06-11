#pragma once

/**
 * VarianceModeler.h - Articulation Variance Modeling and Prominence Curves
 * ========================================================================
 *
 * PRROT/PARROT Voice-Instrument Compiler - Tier C (Embedded Core)
 *
 * RT-safe articulation variance modeling and prominence curve generation.
 */

#include "prrot/VoiceProfile.h"
#include <array>
#include <cstdint>
#include <cstddef>

namespace prrot {

/**
 * VarianceModeler - Articulation variance and prominence modeling
 */
class VarianceModeler {
public:
    VarianceModeler();
    ~VarianceModeler() = default;

    // Model articulation variance for a phoneme
    float modelVariance(
        PhonemeType phoneme,
        const ArticulationVariance& variance_params
    ) const noexcept;

    // Generate prominence curve
    struct ProminenceCurve {
        std::array<float, 256> values; // Prominence values over time
        size_t num_points = 0;
    };

    ProminenceCurve generateProminenceCurve(
        const std::vector<PhonemeType>& phoneme_sequence,
        const ProminenceCharacteristics& prominence_params,
        size_t num_points
    ) const noexcept;

    // Compute articulation consistency score
    float computeConsistencyScore(
        const std::vector<PhonemeType>& phoneme_sequence,
        const ArticulationVariance& variance_params
    ) const noexcept;

private:
    // Helper functions
    float getPhonemeVariance(PhonemeType phoneme, const ArticulationVariance& params) const noexcept;
};

} // namespace prrot
