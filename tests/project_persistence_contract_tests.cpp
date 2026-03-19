#include "project/ProjectManager.h"

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

struct TestCase {
    const char* name;
    bool (*run)();
};

class ScopedEnvVar {
public:
    ScopedEnvVar(const char* name, const char* value)
        : name_(name), hadValue_(false) {
        if (const char* existing = std::getenv(name_); existing != nullptr) {
            hadValue_ = true;
            previousValue_ = existing;
        }

        if (value != nullptr) {
            ::setenv(name_, value, 1);
        } else {
            ::unsetenv(name_);
        }
    }

    ~ScopedEnvVar() {
        if (hadValue_) {
            ::setenv(name_, previousValue_.c_str(), 1);
        } else {
            ::unsetenv(name_);
        }
    }

private:
    const char* name_;
    bool hadValue_;
    std::string previousValue_;
};

bool expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "  FAIL: " << message << std::endl;
        return false;
    }
    return true;
}

kelly::PluginState::Preset makeContractPreset() {
    kelly::PluginState::Preset preset;
    preset.name = "Contract Project";
    preset.description = "project persistence contract";
    preset.author = "test-suite";
    preset.valence = -0.25f;
    preset.arousal = 0.65f;
    preset.intensity = 0.72f;
    preset.complexity = 0.55f;
    preset.humanize = 0.31f;
    preset.feel = -0.10f;
    preset.dynamics = 0.81f;
    preset.bars = 8;
    preset.bypass = false;
    preset.woundDescription = "make it darker";
    preset.selectedEmotionId = 7;
    preset.cassetteState.sideA.description = "side-a";
    preset.cassetteState.sideA.intensity = 0.4f;
    preset.cassetteState.sideA.emotionId = 7;
    preset.cassetteState.sideB.description = "side-b";
    preset.cassetteState.sideB.intensity = 0.8f;
    preset.cassetteState.sideB.emotionId = 12;
    preset.cassetteState.isFlipped = false;
    return preset;
}

kelly::GeneratedMidi makeContractGeneratedMidi() {
    kelly::GeneratedMidi generated;
    generated.tempoBpm = 123;
    generated.bars = 8;
    generated.key = "D";
    generated.mode = "minor";
    generated.bpm = 123.0f;
    generated.lengthInBeats = 32.0;

    kelly::MidiNote melody;
    melody.pitch = 62;
    melody.velocity = 96;
    melody.startTick = 0;
    melody.durationTicks = 480;
    generated.melody = std::vector<kelly::MidiNote>{melody};

    kelly::Chord chord;
    chord.symbol = "Dm";
    chord.root = "D";
    chord.quality = "minor";
    chord.pitches = {62, 65, 69};
    chord.startBeat = 0.0;
    chord.duration = 4.0;
    generated.chords.push_back(chord);

    return generated;
}

bool testUnchangedProjectResaveIsByteStable() {
    ScopedEnvVar clearSourceDateEpoch("SOURCE_DATE_EPOCH", nullptr);

    midikompanion::ProjectManager manager;
    const auto tempDir = juce::File::getCurrentWorkingDirectory().getChildFile("runtime_contract_tmp");
    tempDir.createDirectory();
    const auto token = juce::String::toHexString(juce::Time::getHighResolutionTicks());
    const auto projectFile = tempDir.getChildFile("project_persistence_" + token + ".midikompanion");
    projectFile.deleteFile();

    const auto preset = makeContractPreset();
    const auto generated = makeContractGeneratedMidi();
    const std::vector<kelly::MidiNote> vocalNotes{};
    const std::vector<juce::String> lyrics{juce::String("line one")};
    const std::vector<int> selectedEmotionIds{7, 12};
    const std::optional<int> primaryEmotionId{7};

    if (!expect(manager.saveProject(projectFile, preset, generated, vocalNotes, lyrics,
                                    selectedEmotionIds, primaryEmotionId),
                "first project save should succeed")) {
        projectFile.deleteFile();
        return false;
    }

    const auto firstJson = projectFile.loadFileAsString();
    juce::Thread::sleep(1100);

    if (!expect(manager.saveProject(projectFile, preset, generated, vocalNotes, lyrics,
                                    selectedEmotionIds, primaryEmotionId),
                "unchanged project resave should succeed")) {
        projectFile.deleteFile();
        return false;
    }

    const auto secondJson = projectFile.loadFileAsString();
    projectFile.deleteFile();
    return expect(firstJson == secondJson,
                  "unchanged project resave should preserve byte-identical JSON");
}

bool testSchemaVersionAndSourceDateEpochPropagation() {
    ScopedEnvVar sourceDateEpoch("SOURCE_DATE_EPOCH", "1700000000");

    midikompanion::ProjectManager manager;
    const auto tempDir = juce::File::getCurrentWorkingDirectory().getChildFile("runtime_contract_tmp");
    tempDir.createDirectory();
    const auto token = juce::String::toHexString(juce::Time::getHighResolutionTicks());
    const auto projectFile = tempDir.getChildFile("project_schema_" + token + ".midikompanion");
    projectFile.deleteFile();

    const auto preset = makeContractPreset();
    const auto generated = makeContractGeneratedMidi();
    const std::vector<kelly::MidiNote> vocalNotes{};
    const std::vector<juce::String> lyrics{juce::String("line one")};
    const std::vector<int> selectedEmotionIds{7, 12};
    const std::optional<int> primaryEmotionId{7};

    if (!expect(manager.saveProject(projectFile, preset, generated, vocalNotes, lyrics,
                                    selectedEmotionIds, primaryEmotionId),
                "deterministic timestamp project save should succeed")) {
        projectFile.deleteFile();
        return false;
    }

    if (!expect(manager.isValidProjectFile(projectFile),
                "saved project should satisfy validity checks")) {
        projectFile.deleteFile();
        return false;
    }

    juce::var jsonData;
    const auto parseResult = juce::JSON::parse(projectFile.loadFileAsString(), jsonData);
    if (!expect(parseResult.wasOk() && jsonData.isObject(),
                "saved project JSON should parse into an object")) {
        projectFile.deleteFile();
        return false;
    }

    auto* obj = jsonData.getDynamicObject();
    if (!expect(obj != nullptr, "project JSON should expose a dynamic object")) {
        projectFile.deleteFile();
        return false;
    }

    const auto serialized = juce::JSON::toString(jsonData, true);
    if (!expect(serialized.indexOf("\"schemaVersion\"") >= 0,
                "project JSON should include schemaVersion")) {
        projectFile.deleteFile();
        return false;
    }
    if (!expect(serialized.indexOf("\"schemaVersion\"") < serialized.indexOf("\"version\""),
                "schemaVersion should appear before version")) {
        projectFile.deleteFile();
        return false;
    }
    if (!expect(obj->hasProperty("invariants") && obj->getProperty("invariants").isArray(),
                "project JSON should include invariant list")) {
        projectFile.deleteFile();
        return false;
    }
    if (!expect(obj->getProperty("schemaVersion").toString() == "1.0",
                "project schemaVersion should match expected value")) {
        projectFile.deleteFile();
        return false;
    }

    const juce::int64 expectedMillis = 1700000000LL * 1000LL;
    if (!expect(static_cast<juce::int64>(obj->getProperty("createdTime")) == expectedMillis,
                "project createdTime should honor SOURCE_DATE_EPOCH")) {
        projectFile.deleteFile();
        return false;
    }
    if (!expect(static_cast<juce::int64>(obj->getProperty("modifiedTime")) == expectedMillis,
                "project modifiedTime should honor SOURCE_DATE_EPOCH")) {
        projectFile.deleteFile();
        return false;
    }

    if (!expect(obj->hasProperty("pluginState") && obj->getProperty("pluginState").isObject(),
                "project JSON should include nested pluginState")) {
        projectFile.deleteFile();
        return false;
    }

    auto* pluginStateObj = obj->getProperty("pluginState").getDynamicObject();
    if (!expect(pluginStateObj != nullptr, "pluginState should parse as an object")) {
        projectFile.deleteFile();
        return false;
    }
    if (!expect(pluginStateObj->getProperty("schemaVersion").toString() == "1.0",
                "nested pluginState should emit schemaVersion")) {
        projectFile.deleteFile();
        return false;
    }
    if (!expect(pluginStateObj->hasProperty("invariants") &&
                    pluginStateObj->getProperty("invariants").isArray(),
                "nested pluginState should include invariant list")) {
        projectFile.deleteFile();
        return false;
    }
    if (!expect(static_cast<juce::int64>(pluginStateObj->getProperty("createdTime")) == expectedMillis,
                "nested pluginState createdTime should honor SOURCE_DATE_EPOCH")) {
        projectFile.deleteFile();
        return false;
    }
    if (!expect(static_cast<juce::int64>(pluginStateObj->getProperty("modifiedTime")) == expectedMillis,
                "nested pluginState modifiedTime should honor SOURCE_DATE_EPOCH")) {
        projectFile.deleteFile();
        return false;
    }

    projectFile.deleteFile();
    return true;
}

bool testObsoleteProjectSchemaFailsLoadWithUsefulError() {
    midikompanion::ProjectManager manager;
    const auto tempDir = juce::File::getCurrentWorkingDirectory().getChildFile("runtime_contract_tmp");
    tempDir.createDirectory();
    const auto token = juce::String::toHexString(juce::Time::getHighResolutionTicks());
    const auto projectFile = tempDir.getChildFile("project_obsolete_" + token + ".midikompanion");
    projectFile.deleteFile();

    const auto preset = makeContractPreset();
    const auto generated = makeContractGeneratedMidi();
    if (!expect(manager.saveProject(projectFile, preset, generated),
                "baseline project save should succeed before obsolete-schema mutation")) {
        projectFile.deleteFile();
        return false;
    }

    juce::var jsonData;
    const auto parseResult = juce::JSON::parse(projectFile.loadFileAsString(), jsonData);
    if (!expect(parseResult.wasOk() && jsonData.isObject(),
                "saved project should parse before schema mutation")) {
        projectFile.deleteFile();
        return false;
    }

    auto* obj = jsonData.getDynamicObject();
    obj->setProperty("schemaVersion", "0.9");
    if (!projectFile.replaceWithText(juce::JSON::toString(jsonData, true))) {
        std::cerr << "  FAIL: unable to rewrite obsolete-schema project fixture" << std::endl;
        projectFile.deleteFile();
        return false;
    }

    kelly::PluginState::Preset loadedPreset;
    kelly::GeneratedMidi loadedGenerated;
    std::vector<kelly::MidiNote> vocalNotes;
    std::vector<juce::String> lyrics;
    std::vector<int> selectedEmotionIds;
    std::optional<int> primaryEmotionId;
    const bool loaded = manager.loadProject(projectFile, loadedPreset, loadedGenerated, vocalNotes,
                                            lyrics, selectedEmotionIds, primaryEmotionId);
    const auto error = manager.getLastError();
    projectFile.deleteFile();

    return expect(!loaded, "obsolete project schema should fail load")
        && expect(error.contains("obsolete"), "obsolete schema load error should mention obsolescence");
}

bool testCorruptedProjectInvariantListFailsLoadWithUsefulError() {
    midikompanion::ProjectManager manager;
    const auto tempDir = juce::File::getCurrentWorkingDirectory().getChildFile("runtime_contract_tmp");
    tempDir.createDirectory();
    const auto token = juce::String::toHexString(juce::Time::getHighResolutionTicks());
    const auto projectFile = tempDir.getChildFile("project_corrupt_" + token + ".midikompanion");
    projectFile.deleteFile();

    const auto preset = makeContractPreset();
    const auto generated = makeContractGeneratedMidi();
    if (!expect(manager.saveProject(projectFile, preset, generated),
                "baseline project save should succeed before invariant mutation")) {
        projectFile.deleteFile();
        return false;
    }

    juce::var jsonData;
    const auto parseResult = juce::JSON::parse(projectFile.loadFileAsString(), jsonData);
    if (!expect(parseResult.wasOk() && jsonData.isObject(),
                "saved project should parse before invariant mutation")) {
        projectFile.deleteFile();
        return false;
    }

    auto* obj = jsonData.getDynamicObject();
    juce::Array<juce::var> invalidInvariants;
    invalidInvariants.add(juce::String("schemaVersion_required"));
    obj->setProperty("invariants", juce::var(invalidInvariants));
    if (!projectFile.replaceWithText(juce::JSON::toString(jsonData, true))) {
        std::cerr << "  FAIL: unable to rewrite corrupted project fixture" << std::endl;
        projectFile.deleteFile();
        return false;
    }

    kelly::PluginState::Preset loadedPreset;
    kelly::GeneratedMidi loadedGenerated;
    std::vector<kelly::MidiNote> vocalNotes;
    std::vector<juce::String> lyrics;
    std::vector<int> selectedEmotionIds;
    std::optional<int> primaryEmotionId;
    const bool loaded = manager.loadProject(projectFile, loadedPreset, loadedGenerated, vocalNotes,
                                            lyrics, selectedEmotionIds, primaryEmotionId);
    const auto error = manager.getLastError();
    projectFile.deleteFile();

    return expect(!loaded, "corrupted invariant list should fail load")
        && expect(error.contains("corrupted"), "corrupted invariant load error should mention corruption");
}

} // namespace

int main() {
    const std::vector<TestCase> tests = {
        {"unchanged project resave is byte-stable", testUnchangedProjectResaveIsByteStable},
        {"schemaVersion and SOURCE_DATE_EPOCH propagate", testSchemaVersionAndSourceDateEpochPropagation},
        {"obsolete project schema fails with useful error", testObsoleteProjectSchemaFailsLoadWithUsefulError},
        {"corrupted invariant list fails with useful error", testCorruptedProjectInvariantListFailsLoadWithUsefulError},
    };

    int failed = 0;
    for (const auto& test : tests) {
        std::cout << "[TEST] " << test.name << std::endl;
        if (!test.run()) {
            ++failed;
        }
    }

    if (failed > 0) {
        std::cerr << failed << " project persistence contract test(s) failed." << std::endl;
        return 1;
    }

    std::cout << "All project persistence contract tests passed." << std::endl;
    return 0;
}
