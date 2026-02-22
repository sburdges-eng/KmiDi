#include "prrot/VarianceModeler.h"
#include "penta/common/RTLogger.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace prrot {

VarianceModeler::VarianceModeler() {
}

float VarianceModeler::modelVariance(
    PhonemeType phoneme,
    const ArticulationVariance& variance_params
) const noexcept {
    float base_variance = variance_params.variance_range_min +
                         (variance_params.variance_range_max - variance_params.variance_range_min) * 0.5f;

    // Get phoneme-specific variance
    float phoneme_variance = getPhonemeVariance(phoneme, variance_params);

    // Combine base and phoneme-specific variance
    return base_variance * (1.0f + phoneme_variance);
}

VarianceModeler::ProminenceCurve VarianceModeler::generateProminenceCurve(
    const std::vector<PhonemeType>& phoneme_sequence,
    const ProminenceCharacteristics& prominence_params,
    size_t num_points
) const noexcept {
    ProminenceCurve curve;

    // Validate inputs
    if (phoneme_sequence.empty()) {
        penta::getLogger().logRT(penta::LogLevel::Warning,
            "VarianceModeler::generateProminenceCurve: Empty phoneme sequence");
        return curve;
    }

    if (num_points == 0) {
        penta::getLogger().logRT(penta::LogLevel::Warning,
            "VarianceModeler::generateProminenceCurve: num_points is zero");
        return curve;
    }

    // Validate prominence params
    if (prominence_params.stress_amplitude_multiplier < 0.0f ||
        prominence_params.stress_duration_multiplier < 0.0f ||
        prominence_params.emphasis_probability < 0.0f ||
        prominence_params.emphasis_probability > 1.0f) {
        penta::getLogger().logRT(penta::LogLevel::Warning,
            "VarianceModeler::generateProminenceCurve: Invalid prominence parameters");
        return curve;
    }

    curve.num_points = std::min(num_points, curve.values.size());

    // Generate prominence values based on phoneme types and stress patterns
    // Vowels typically have higher prominence than consonants
    // Stressed phonemes have higher prominence

    size_t phonemes_per_point = phoneme_sequence.size() / curve.num_points;
    if (phonemes_per_point == 0) {
        phonemes_per_point = 1;
    }

    for (size_t i = 0; i < curve.num_points; ++i) {
        size_t phoneme_idx = i * phonemes_per_point;
        if (phoneme_idx < phoneme_sequence.size()) {
            PhonemeType phoneme = phoneme_sequence[phoneme_idx];

            // Base prominence: vowels > consonants
            float base_prominence = isVowel(phoneme) ? 0.7f : 0.3f;

            // Apply prominence characteristics
            base_prominence *= prominence_params.stress_amplitude_multiplier;

            // Add variance
            float variance = 0.2f * prominence_params.emphasis_probability;
            base_prominence += variance * (static_cast<float>(rand() % 100) / 100.0f - 0.5f);

            curve.values[i] = std::clamp(base_prominence, 0.0f, 1.0f);
        } else {
            curve.values[i] = 0.5f;
        }
    }

    return curve;
}

float VarianceModeler::computeConsistencyScore(
    const std::vector<PhonemeType>& phoneme_sequence,
    const ArticulationVariance& variance_params
) const noexcept {
    if (phoneme_sequence.empty()) {
        return 0.0f;
    }

    // Count occurrences of each phoneme
    std::map<PhonemeType, size_t> phoneme_counts;
    for (PhonemeType phoneme : phoneme_sequence) {
        phoneme_counts[phoneme]++;
    }

    // Compute consistency: lower variance in articulation = higher consistency
    float total_variance = 0.0f;
    for (const auto& pair : phoneme_counts) {
        float phoneme_variance = getPhonemeVariance(pair.first, variance_params);
        total_variance += phoneme_variance * static_cast<float>(pair.second);
    }

    float avg_variance = total_variance / static_cast<float>(phoneme_sequence.size());

    // Consistency score: 1.0 = very consistent, 0.0 = very variable
    float consistency = 1.0f - std::min(avg_variance, 1.0f);

    // Combine with overall consistency score
    return consistency * variance_params.consistency_score;
}

float VarianceModeler::getPhonemeVariance(PhonemeType phoneme, const ArticulationVariance& params) const noexcept {
    auto it = params.per_phoneme_variance.find(static_cast<uint8_t>(phoneme));
    if (it != params.per_phoneme_variance.end()) {
        return it->second;
    }

    // Default variance based on phoneme type
    if (isVowel(phoneme)) {
        return 0.1f; // Vowels typically more consistent
    } else if (isConsonant(phoneme)) {
        return 0.2f; // Consonants may have more variance
    }

    return 0.15f; // Default
}

} // namespace prrot
