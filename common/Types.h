#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace kelly {

//=============================================================================
// Emotion representation
//=============================================================================

struct Emotion {
    std::string name;
    float valence = 0.0f; // -1.0 to 1.0
    float arousal = 0.0f; // 0.0 to 1.0
};

//=============================================================================
// Input types
//=============================================================================

struct Wound {
    std::string description;
    float intensity = 0.0f;
    std::string source;
};

struct SideA {
    std::string description;
    float intensity = 0.0f;
    std::optional<std::string> additional;
};

struct SideB {
    std::string description;
    float intensity = 0.0f;
    std::optional<std::string> additional;
};

struct PerformanceDelta {
    int noteOnCount = 0;
    int noteOffCount = 0;
    double durationBeats = 0.0;

    bool isMeaningful() const {
        return noteOnCount > 0 || noteOffCount > 0 || durationBeats > 0.0;
    }
};

struct ParameterDelta {
    std::string id;
    float value = 0.0f;

    bool isMeaningful() const {
        return !id.empty();
    }
};

struct UserInput {
    std::string text;
    std::optional<PerformanceDelta> performanceDelta = std::nullopt;
    std::optional<ParameterDelta> parameterDelta = std::nullopt;
    std::unordered_map<std::string, float> axisOverrides;

    bool hasTextDelta() const {
        return text.find_first_not_of(" \t\n\r") != std::string::npos;
    }

    bool hasPerformanceDelta() const {
        return performanceDelta.has_value() && performanceDelta->isMeaningful();
    }

    bool hasParameterDelta() const {
        return parameterDelta.has_value() && parameterDelta->isMeaningful();
    }

    bool hasAnyDelta() const {
        return hasTextDelta() || hasPerformanceDelta() || hasParameterDelta();
    }
};

//=============================================================================
// Scope + result types
//=============================================================================

enum class AbstainReason : uint8_t {
    NONE = 0,
    EMPTY_INPUT,
    UNDERSPECIFIED,
    GENERATOR_EMPTY,
    NON_AUDITABLE_PATH
};

enum class IntentResultStatus : uint8_t {
    VALID = 0,
    ABSTAIN = 1,
    INVALID = 2
};

enum class DomainGroup : uint32_t {
    NONE = 0,
    HARMONY = 1u << 0,
    RHYTHM = 1u << 1,
    TEXTURE = 1u << 2,
    MELODY = 1u << 3,
    PLAYBACK = 1u << 4
};

inline uint32_t mask(DomainGroup g) noexcept {
    return static_cast<uint32_t>(g);
}

inline DomainGroup operator|(DomainGroup a, DomainGroup b) noexcept {
    return static_cast<DomainGroup>(mask(a) | mask(b));
}

inline DomainGroup operator&(DomainGroup a, DomainGroup b) noexcept {
    return static_cast<DomainGroup>(mask(a) & mask(b));
}

struct RuleBreak {
    std::string rule;
    std::string reason;
};

struct SuggestedVariant {
    std::string description;
    uint32_t changedGroupsMask = 0;
};

struct SuggestedVariants {
    std::vector<SuggestedVariant> items;
};

struct IntentResult {
    IntentResultStatus status = IntentResultStatus::VALID;
    Emotion emotion;
    std::string mode;
    float tempo = 1.0f;
    std::vector<RuleBreak> ruleBreaks;
    bool escalationTokenPresent = false;
    AbstainReason abstainReason = AbstainReason::NONE;
    uint32_t requestedGroupsMask = 0;
    uint32_t modifiedGroupsMask = 0;
    uint32_t leakedGroupsMask = 0;
    SuggestedVariants alternatives;

    bool isValid() const { return status == IntentResultStatus::VALID; }
    bool isAbstain() const { return status == IntentResultStatus::ABSTAIN; }
    bool isInvalid() const { return status == IntentResultStatus::INVALID; }

    static IntentResult abstain(AbstainReason reason = AbstainReason::NONE) {
        IntentResult r;
        r.status = IntentResultStatus::ABSTAIN;
        r.abstainReason = reason;
        return r;
    }

    static IntentResult invalid() {
        IntentResult r;
        r.status = IntentResultStatus::INVALID;
        return r;
    }
};

inline bool canProduceOutput(const IntentResult& result) {
    return result.status == IntentResultStatus::VALID;
}

//=============================================================================
// Minimal parameter space placeholder (keeps legacy compile surface stable)
//=============================================================================

struct AxisValue {
    enum class Source : uint8_t { DEFAULT = 0, INFERRED = 1, USER_EXPLICIT = 2, CONSTRAINT = 3 };
    float value = 0.0f;
    Source source = Source::DEFAULT;
};

struct AxisDescriptor {
    std::string id;
    std::string group;
    float minValue = 0.0f;
    float maxValue = 1.0f;
    float defaultValue = 0.0f;
    bool escalationGated = false;
};

class ParameterSpace {
public:
    const AxisValue& get(const std::string& id) const {
        auto it = values_.find(id);
        if (it == values_.end()) {
            static const AxisValue kDefault{};
            return kDefault;
        }
        return it->second;
    }

    void set(const std::string& id, float value, AxisValue::Source source = AxisValue::Source::USER_EXPLICIT) {
        values_[id] = AxisValue{value, source};
    }

    void setFromMap(const std::unordered_map<std::string, float>& updates, AxisValue::Source source) {
        for (const auto& [id, value] : updates) {
            set(id, value, source);
        }
    }

    std::unordered_map<std::string, AxisValue> snapshot() const { return values_; }
    void enforceRanges() {}
    void enforceEscalationGate(bool) {}
    size_t axisCount() const { return values_.size(); }
    std::vector<const AxisDescriptor*> allAxes() const { return {}; }
    const AxisDescriptor* findAxis(const std::string&) const { return nullptr; }

private:
    std::unordered_map<std::string, AxisValue> values_;
};

inline void registerSeedAxes(ParameterSpace&) {}

//=============================================================================
// MIDI generation types
//=============================================================================

struct MidiNote {
    int pitch = 60;
    int velocity = 80;
    double startBeat = 0.0;
    double duration = 1.0;
};

struct Chord {
    std::vector<int> pitches;
    double startBeat = 0.0;
    double duration = 1.0;
};

struct GeneratedMidi {
    std::vector<Chord> chords;
    std::optional<std::vector<int>> chordVelocities = std::nullopt;
    std::optional<std::vector<MidiNote>> melody = std::nullopt;
    std::optional<std::vector<MidiNote>> bass = std::nullopt;
    float bpm = 120.0f;
    double lengthInBeats = 16.0;
};

inline int clampMidiPitch(int pitch) {
    return pitch < 0 ? 0 : (pitch > 127 ? 127 : pitch);
}

inline int clampMidiVelocity(int velocity) {
    return velocity < 1 ? 1 : (velocity > 127 ? 127 : velocity);
}

static constexpr size_t kMaxPcmBufferSamples = 96000 * 60 * 10;
static constexpr double kMaxLengthInBeats = 1024.0;

//=============================================================================
// Rule bits + validation reports (R1..R5)
//=============================================================================

enum RuleBits : uint8_t {
    R1_EMPTY_INPUT = 1u << 0,
    R2_UNDERSPECIFIED = 1u << 1,
    R3_PRECEDENCE = 1u << 2,
    R4_ESCALATION = 1u << 3,
    R5_SCOPE_ISOLATION = 1u << 4
};

struct GuardrailValidationReport {
    uint8_t rulesEvaluated = 5;
    uint8_t ruleResults = 0;
    IntentResultStatus finalStatus = IntentResultStatus::ABSTAIN;
    AbstainReason abstainReason = AbstainReason::NONE;
    std::vector<std::string> violations;
    uint32_t requestedGroupsMask = 0;
    uint32_t modifiedGroupsMask = 0;
    uint32_t leakedGroupsMask = 0;
};

class GuardrailValidator;
class OutputPlanValidator;
class UserActionAuthority;

class ValidatedIntent {
public:
    ValidatedIntent(ValidatedIntent&&) noexcept = default;
    ValidatedIntent& operator=(ValidatedIntent&&) noexcept = default;
    ValidatedIntent(const ValidatedIntent&) = delete;
    ValidatedIntent& operator=(const ValidatedIntent&) = delete;

    const IntentResult& intent() const { return intent_; }
    const std::string& mode() const { return intent_.mode; }
    float tempo() const { return intent_.tempo; }
    bool escalationTokenPresent() const { return intent_.escalationTokenPresent; }

private:
    friend class GuardrailValidator;
    explicit ValidatedIntent(IntentResult&& intent) : intent_(std::move(intent)) {}
    IntentResult intent_;
};

struct OutputLayerAuthorization {
    bool chordsAuthorized = true;
    bool melodyAuthorized = false;
    bool bassAuthorized = false;
    bool velocitiesAuthorized = false;
    bool rhythmDensityIncreaseAuthorized = false;
    uint8_t maxVoiceCount = 3;
    size_t maxChordCount = 0;
    double maxRhythmicDensity = 0.0;
};

class ValidatedOutputPlan {
public:
    ValidatedOutputPlan(ValidatedOutputPlan&&) noexcept = default;
    ValidatedOutputPlan& operator=(ValidatedOutputPlan&&) noexcept = default;
    ValidatedOutputPlan(const ValidatedOutputPlan&) = delete;
    ValidatedOutputPlan& operator=(const ValidatedOutputPlan&) = delete;

    const GeneratedMidi& midi() const { return midi_; }
    const OutputLayerAuthorization& authorization() const { return auth_; }

private:
    friend class OutputPlanValidator;
    ValidatedOutputPlan(GeneratedMidi&& midi, OutputLayerAuthorization auth)
        : midi_(std::move(midi)), auth_(std::move(auth)) {}

    GeneratedMidi midi_;
    OutputLayerAuthorization auth_;
};

struct OutputValidationResult {
    std::optional<ValidatedOutputPlan> validated;
    std::vector<std::string> strippedLayers;
};

class PlaybackAuthorizationToken {
public:
    PlaybackAuthorizationToken(PlaybackAuthorizationToken&&) noexcept = default;
    PlaybackAuthorizationToken& operator=(PlaybackAuthorizationToken&&) noexcept = default;
    PlaybackAuthorizationToken(const PlaybackAuthorizationToken&) = delete;
    PlaybackAuthorizationToken& operator=(const PlaybackAuthorizationToken&) = delete;

    bool isAuthorized() const noexcept { return authorized_; }

private:
    friend class UserActionAuthority;
    explicit PlaybackAuthorizationToken(bool authorized) : authorized_(authorized) {}
    bool authorized_ = false;
};

class UserActionAuthority {
public:
    static PlaybackAuthorizationToken authorizePlayback(bool explicitUserAction = false) {
        return PlaybackAuthorizationToken(explicitUserAction);
    }
};

} // namespace kelly
