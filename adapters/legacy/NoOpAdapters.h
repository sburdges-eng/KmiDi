#pragma once

#include "interfaces/IArrangementEngine.h"
#include "interfaces/IBiometricInput.h"
#include "interfaces/IHarmonyEngine.h"
#include "interfaces/IIntentEnricher.h"
#include "interfaces/IGrooveEngine.h"
#include "interfaces/IMLAssistService.h"
#include "interfaces/IVoiceLayer.h"
#include <optional>

namespace kelly {

class NoOpIntentEnricher final : public IIntentEnricher {
public:
    const char* id() const override { return "noop.intent_enricher"; }
    void enrich(const UserInput&, IntentResult&) override {}
};

class NoOpHarmonyEngine final : public IHarmonyEngine {
public:
    const char* id() const override { return "noop.harmony_engine"; }
    std::optional<std::vector<Chord>> generateChords(const IntentResult&) override { return std::nullopt; }
};

class NoOpGrooveEngine final : public IGrooveEngine {
public:
    const char* id() const override { return "noop.groove_engine"; }
    void applyGroove(const IntentResult&, GeneratedMidi&) override {}
};

class NoOpArrangementEngine final : public IArrangementEngine {
public:
    const char* id() const override { return "noop.arrangement_engine"; }
    void arrange(const IntentResult&, GeneratedMidi&) override {}
};

class NoOpMLAssistService final : public IMLAssistService {
public:
    const char* id() const override { return "noop.ml_assist"; }
    std::optional<IntentResult> inferBias(const UserInput&) override { return std::nullopt; }
};

class NoOpVoiceLayer final : public IVoiceLayer {
public:
    const char* id() const override { return "noop.voice_layer"; }
    bool isAvailable() const override { return false; }
    void reset() override {}
};

class NoOpBiometricInput final : public IBiometricInput {
public:
    const char* id() const override { return "noop.biometric_input"; }
    std::optional<BiometricSnapshot> readSnapshot() override { return std::nullopt; }
};

} // namespace kelly
