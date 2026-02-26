#pragma once

#include "common/Types.h"

#include <optional>
#include <string>

namespace kelly {

struct GuardrailValidationResult {
    std::optional<ValidatedIntent> validated;
    GuardrailValidationReport report;
};

struct GuardrailInputContext {
    GuardrailInputContext() = delete;
    GuardrailInputContext(bool hasAny,
                          bool hasText,
                          bool hasPerformance,
                          bool hasParameter,
                          uint32_t requestedMask,
                          std::string normalizedTextValue)
        : hasAnyDelta(hasAny),
          hasTextDelta(hasText),
          hasPerformanceDelta(hasPerformance),
          hasParameterDelta(hasParameter),
          requestedGroupsMask(requestedMask),
          normalizedText(std::move(normalizedTextValue)) {}

    bool hasAnyDelta = false;
    bool hasTextDelta = false;
    bool hasPerformanceDelta = false;
    bool hasParameterDelta = false;
    uint32_t requestedGroupsMask = 0;
    std::string normalizedText;
};

class GuardrailValidator {
public:
    static GuardrailValidationResult validate(IntentResult&& intent, const GuardrailInputContext& context) {
        GuardrailValidationResult result;
        auto& report = result.report;
        report.finalStatus = intent.status;
        report.abstainReason = intent.abstainReason;
        report.requestedGroupsMask = context.requestedGroupsMask;
        report.modifiedGroupsMask = intent.modifiedGroupsMask;
        report.leakedGroupsMask = intent.modifiedGroupsMask & (~context.requestedGroupsMask);

        const bool r1Valid = context.hasAnyDelta
            ? !(intent.status == IntentResultStatus::VALID && intent.abstainReason == AbstainReason::EMPTY_INPUT)
            : (intent.status == IntentResultStatus::ABSTAIN && intent.abstainReason == AbstainReason::EMPTY_INPUT);
        if (r1Valid) {
            report.ruleResults |= R1_EMPTY_INPUT;
        } else {
            report.violations.push_back("R1_EMPTY_INPUT_CONFLICT");
            report.finalStatus = IntentResultStatus::INVALID;
        }

        const bool r2Valid = context.hasTextDelta
            ? !(intent.status == IntentResultStatus::VALID && intent.abstainReason == AbstainReason::UNDERSPECIFIED)
            : (intent.status != IntentResultStatus::VALID);
        if (r2Valid) {
            report.ruleResults |= R2_UNDERSPECIFIED;
        } else {
            report.violations.push_back("R2_UNDERSPECIFIED_CONFLICT");
            report.finalStatus = IntentResultStatus::INVALID;
        }

        if (intent.requestedGroupsMask != context.requestedGroupsMask) {
            report.violations.push_back("CTX_REQUESTED_GROUPS_MISMATCH");
            report.finalStatus = IntentResultStatus::INVALID;
        }

        report.ruleResults |= R3_PRECEDENCE;

        const bool melodyModified = (intent.modifiedGroupsMask & mask(DomainGroup::MELODY)) != 0;
        const bool melodyRequested = (context.requestedGroupsMask & mask(DomainGroup::MELODY)) != 0;
        const bool r4Valid = intent.escalationTokenPresent || !melodyModified || melodyRequested;
        if (r4Valid) {
            report.ruleResults |= R4_ESCALATION;
        } else {
            report.violations.push_back("R4_ESCALATION_REQUIRED");
            report.finalStatus = IntentResultStatus::INVALID;
        }

        const bool r5Valid = (report.leakedGroupsMask == 0);
        if (r5Valid) {
            report.ruleResults |= R5_SCOPE_ISOLATION;
        } else {
            report.violations.push_back("R5_SCOPE_LEAK");
            report.finalStatus = IntentResultStatus::INVALID;
        }

        if (intent.status == IntentResultStatus::INVALID) {
            report.finalStatus = IntentResultStatus::INVALID;
            if (report.violations.empty()) {
                report.violations.push_back("INPUT_INVALID");
            }
            return result;
        }

        if (intent.status == IntentResultStatus::ABSTAIN) {
            report.finalStatus = IntentResultStatus::ABSTAIN;
            return result;
        }

        if (report.finalStatus == IntentResultStatus::VALID) {
            result.validated = ValidatedIntent(std::move(intent));
        }
        return result;
    }
};

class OutputPlanValidator {
public:
    static OutputValidationResult validate(GeneratedMidi&& midi, const OutputLayerAuthorization& auth) {
        OutputValidationResult result;

        if (!auth.chordsAuthorized && !midi.chords.empty()) {
            return result;
        }

        if (!auth.melodyAuthorized && midi.melody.has_value()) {
            midi.melody = std::nullopt;
            result.strippedLayers.push_back("melody");
        }
        if (!auth.bassAuthorized && midi.bass.has_value()) {
            midi.bass = std::nullopt;
            result.strippedLayers.push_back("bass");
        }
        if (!auth.velocitiesAuthorized && midi.chordVelocities.has_value()) {
            midi.chordVelocities = std::nullopt;
            result.strippedLayers.push_back("chordVelocities");
        }

        for (auto& chord : midi.chords) {
            if (chord.pitches.size() > static_cast<size_t>(auth.maxVoiceCount)) {
                std::sort(chord.pitches.begin(), chord.pitches.end());
                chord.pitches.resize(static_cast<size_t>(auth.maxVoiceCount));
                result.strippedLayers.push_back("voiceClamp");
            }
        }

        if (!auth.rhythmDensityIncreaseAuthorized && auth.maxChordCount > 0 && midi.chords.size() > auth.maxChordCount) {
            midi.chords.resize(auth.maxChordCount);
            result.strippedLayers.push_back("chordCountCap");
        }

        if (!auth.rhythmDensityIncreaseAuthorized && auth.maxRhythmicDensity > 0.0) {
            auto density = [](const GeneratedMidi& m) {
                if (m.lengthInBeats <= 0.0) {
                    return 0.0;
                }
                size_t notes = 0;
                for (const auto& chord : m.chords) {
                    notes += chord.pitches.size();
                }
                if (m.melody.has_value()) {
                    notes += m.melody->size();
                }
                if (m.bass.has_value()) {
                    notes += m.bass->size();
                }
                return static_cast<double>(notes) / m.lengthInBeats;
            };

            while (!midi.chords.empty() && density(midi) > auth.maxRhythmicDensity) {
                midi.chords.pop_back();
                result.strippedLayers.push_back("densityCap");
            }
        }

        result.validated = ValidatedOutputPlan(std::move(midi), auth);
        return result;
    }
};

} // namespace kelly
