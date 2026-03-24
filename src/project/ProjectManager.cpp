#include "project/ProjectManager.h"
#include "common/PersistenceTimestamp.h"
#include "common/Types.h"
#include "plugin/PluginState.h" // Include PluginState for Preset type
#include <array>
#include <cstdlib>
#include <juce_core/juce_core.h>

namespace {

constexpr auto kProjectSchemaVersion = "1.0";
constexpr std::array<const char*, 5> kProjectInvariants = {
    "schemaVersion_required",
    "canonical_key_order",
    "timestamp_stable_when_payload_unchanged",
    "generated_midi_track_order_preserved",
    "selected_emotion_order_preserved",
};

template <size_t N>
juce::Array<juce::var> buildInvariantArray(const std::array<const char*, N>& invariants) {
  juce::Array<juce::var> values;
  for (const auto* invariant : invariants) {
    values.add(juce::String(invariant));
  }
  return values;
}

void appendInvariantTree(juce::ValueTree& parent,
                         const std::array<const char*, 5>& invariants) {
  auto invariantsTree = juce::ValueTree("Invariants");
  for (const auto* invariant : invariants) {
    auto invariantTree = juce::ValueTree("Invariant");
    invariantTree.setProperty("name", juce::String(invariant), nullptr);
    invariantsTree.appendChild(invariantTree, nullptr);
  }
  parent.appendChild(invariantsTree, nullptr);
}

using kelly::persistence::zeroTime;
using kelly::persistence::resolvePersistenceTimestamp;

midikompanion::ProjectManager::ProjectData normalizedProjectForComparison(
    const midikompanion::ProjectManager::ProjectData& projectData) {
  auto normalized = projectData;
  normalized.createdTime = zeroTime();
  normalized.modifiedTime = zeroTime();
  normalized.pluginState.createdTime = zeroTime();
  normalized.pluginState.modifiedTime = zeroTime();
  return normalized;
}

bool projectEquivalentIgnoringTimestamps(
    const midikompanion::ProjectManager::ProjectData& lhs,
    const midikompanion::ProjectManager::ProjectData& rhs) {
  return juce::JSON::toString(normalizedProjectForComparison(lhs).toJson(), false)
      == juce::JSON::toString(normalizedProjectForComparison(rhs).toJson(), false);
}

std::optional<midikompanion::ProjectManager::ProjectData> loadExistingProjectData(
    const juce::File& file) {
  if (!file.existsAsFile()) {
    return std::nullopt;
  }

  juce::var jsonData;
  const auto result = juce::JSON::parse(file.loadFileAsString(), jsonData);
  if (result.failed()) {
    return std::nullopt;
  }

  return midikompanion::ProjectManager::ProjectData::fromJson(jsonData);
}

void applyProjectPersistenceTimestamps(
    midikompanion::ProjectManager::ProjectData& projectData,
    const std::optional<midikompanion::ProjectManager::ProjectData>& existingProject) {
  const auto resolvedTimestamp = resolvePersistenceTimestamp();

  if (!existingProject.has_value()) {
    projectData.createdTime = resolvedTimestamp;
    projectData.modifiedTime = resolvedTimestamp;
    projectData.pluginState.createdTime = resolvedTimestamp;
    projectData.pluginState.modifiedTime = resolvedTimestamp;
    return;
  }

  projectData.createdTime = existingProject->createdTime.toMilliseconds() > 0
      ? existingProject->createdTime
      : resolvedTimestamp;
  projectData.pluginState.createdTime = existingProject->pluginState.createdTime.toMilliseconds() > 0
      ? existingProject->pluginState.createdTime
      : projectData.createdTime;

  if (projectEquivalentIgnoringTimestamps(*existingProject, projectData)) {
    projectData.modifiedTime = existingProject->modifiedTime.toMilliseconds() > 0
        ? existingProject->modifiedTime
        : projectData.createdTime;
    projectData.pluginState.modifiedTime =
        existingProject->pluginState.modifiedTime.toMilliseconds() > 0
        ? existingProject->pluginState.modifiedTime
        : projectData.pluginState.createdTime;
    return;
  }

  projectData.modifiedTime = resolvedTimestamp;
  projectData.pluginState.modifiedTime = resolvedTimestamp;
}

bool hasSupportedProjectSchema(const juce::DynamicObject* obj) {
  if (obj == nullptr) {
    return false;
  }
  if (obj->hasProperty("schemaVersion")) {
    return obj->getProperty("schemaVersion").toString() == kProjectSchemaVersion;
  }
  return obj->hasProperty("version");
}

bool hasSupportedProjectSchema(const juce::ValueTree& tree) {
  if (tree.hasProperty("schemaVersion")) {
    return tree.getProperty("schemaVersion").toString() == kProjectSchemaVersion;
  }
  return tree.hasProperty("version");
}

template <typename T>
bool hasOptionalItems(const std::optional<std::vector<T>>& items) {
  return items.has_value() && !items->empty();
}

template <typename T>
std::vector<T>& ensureOptionalItems(std::optional<std::vector<T>>& items) {
  if (!items.has_value()) {
    items.emplace();
  }
  return *items;
}

} // namespace

namespace midikompanion {

ProjectManager::ProjectManager() { clearError(); }

//==============================================================================
// Project Save/Load Interface
//==============================================================================

bool ProjectManager::saveProject(const juce::File &file,
                                 const kelly::PluginState::Preset &state,
                                 const GeneratedMidi &generatedMidi,
                                 const std::vector<MidiNote> &vocalNotes,
                                 const std::vector<juce::String> &lyrics,
                                 const std::vector<int> &selectedEmotionIds,
                                 const std::optional<int> &primaryEmotionId) {
#if KMIDI_ENABLE_S3
  if (file.getFullPathName().startsWith("s3://")) {
    return saveProjectToS3(file.getFullPathName(), state, generatedMidi,
                            vocalNotes, lyrics, selectedEmotionIds, primaryEmotionId);
  }
#endif

  clearError();
  const auto existingProject = loadExistingProjectData(file);

  // Create project data structure
  ProjectData projectData;
  projectData.name = file.getFileNameWithoutExtension();
  projectData.versionMajor = 1;
  projectData.versionMinor = 0;

  // Copy plugin state (full preset including CassetteState)
  projectData.pluginState = state;

  // Copy generated MIDI
  projectData.generatedMidi = generatedMidi;

  // Copy vocal data
  projectData.vocalNotes = vocalNotes;
  projectData.lyrics = lyrics;

  // Copy emotion selections
  projectData.selectedEmotionIds = selectedEmotionIds;
  projectData.primaryEmotionId = primaryEmotionId;

  // Extract tempo from generated MIDI or plugin state
  if (generatedMidi.bpm > 0.0f) {
    projectData.tempo = generatedMidi.bpm;
  } else {
    projectData.tempo = 120.0f; // Default
  }

  applyProjectPersistenceTimestamps(projectData, existingProject);

  // Convert to JSON
  juce::var json = projectData.toJson();

  // Write to file
  juce::String jsonString = juce::JSON::toString(json, true);
  bool success = file.replaceWithText(jsonString);

  if (!success) {
    setError("Failed to write project file: " + file.getFullPathName());
    return false;
  }

  return true;
}

bool ProjectManager::loadProject(const juce::File &file,
                                 kelly::PluginState::Preset &outState,
                                 GeneratedMidi &outGeneratedMidi,
                                 std::vector<MidiNote> &outVocalNotes,
                                 std::vector<juce::String> &outLyrics,
                                 std::vector<int> &outSelectedEmotionIds,
                                 std::optional<int> &outPrimaryEmotionId) {
#if KMIDI_ENABLE_S3
  if (file.getFullPathName().startsWith("s3://")) {
    return loadProjectFromS3(file.getFullPathName(), outState, outGeneratedMidi,
                              outVocalNotes, outLyrics, outSelectedEmotionIds, outPrimaryEmotionId);
  }
#endif

  clearError();

  if (!file.existsAsFile()) {
    setError("Project file does not exist: " + file.getFullPathName());
    return false;
  }

  juce::var jsonData;
  juce::Result result = juce::JSON::parse(file.loadFileAsString(), jsonData);
  if (result.failed()) {
    setError("Failed to parse project file: " + result.getErrorMessage());
    return false;
  }

  auto projectData = ProjectData::fromJson(jsonData);
  if (!projectData.has_value()) {
    setError("Invalid project file format");
    return false;
  }

  const auto invariantCheck = projectData->verifyInvariants();
  if (!invariantCheck.ok) {
    setError(invariantCheck.message);
    return false;
  }

  outState = projectData->pluginState;
  outGeneratedMidi = projectData->generatedMidi;
  outVocalNotes = projectData->vocalNotes;
  outLyrics = projectData->lyrics;
  outSelectedEmotionIds = projectData->selectedEmotionIds;
  outPrimaryEmotionId = projectData->primaryEmotionId;

  return true;
}

#if KMIDI_ENABLE_S3
bool ProjectManager::loadProjectFromS3(const juce::String& s3Uri,
                                       kelly::PluginState::Preset& outState,
                                       GeneratedMidi& outGeneratedMidi,
                                       std::vector<MidiNote>& outVocalNotes,
                                       std::vector<juce::String>& outLyrics,
                                       std::vector<int>& outSelectedEmotionIds,
                                       std::optional<int>& outPrimaryEmotionId) {
  clearError();
  if (!s3Uri.startsWith("s3://")) {
    setError("Invalid S3 URI: " + s3Uri);
    return false;
  }

  jassertfalse; // S3 stub reached — implementation required
  setError("S3 project loading not yet implemented for current build: " + s3Uri);
  return false;
}

bool ProjectManager::saveProjectToS3(const juce::String& s3Uri,
                                     const kelly::PluginState::Preset& state,
                                     const GeneratedMidi& generatedMidi,
                                     const std::vector<MidiNote>& vocalNotes,
                                     const std::vector<juce::String>& lyrics,
                                     const std::vector<int>& selectedEmotionIds,
                                     const std::optional<int>& primaryEmotionId) {
  clearError();
  if (!s3Uri.startsWith("s3://")) {
    setError("Invalid S3 URI: " + s3Uri);
    return false;
  }

  jassertfalse; // S3 stub reached — implementation required
  setError("S3 project saving not yet implemented for current build: " + s3Uri);
  return false;
}
#endif

bool ProjectManager::isValidProjectFile(const juce::File &file) const {
  if (!file.existsAsFile()) {
    return false;
  }

  // Check file extension
  if (!file.hasFileExtension("midikompanion")) {
    return false;
  }

  // Try to parse as JSON
  juce::var jsonData;
  juce::Result result = juce::JSON::parse(file.loadFileAsString(), jsonData);
  if (result.failed()) {
    return false;
  }

  // Check for required fields
  if (!jsonData.isObject()) {
    return false;
  }

  auto projectData = ProjectData::fromJson(jsonData);
  return projectData.has_value() && projectData->verifyInvariants().ok;
}

std::optional<ProjectManager::ProjectData>
ProjectManager::getProjectMetadata(const juce::File &file) const {
  if (!file.existsAsFile()) {
    return std::nullopt;
  }

  juce::var jsonData;
  juce::Result result = juce::JSON::parse(file.loadFileAsString(), jsonData);
  if (result.failed()) {
    return std::nullopt;
  }

  auto projectData = ProjectData::fromJson(jsonData);
  if (!projectData.has_value()) {
    return std::nullopt;
  }
  if (!projectData->verifyInvariants().ok) {
    return std::nullopt;
  }

  // Return metadata only (don't load full MIDI data)
  ProjectData metadata;
  metadata.schemaVersion = projectData->schemaVersion;
  metadata.invariants = projectData->invariants;
  metadata.name = projectData->name;
  metadata.createdTime = projectData->createdTime;
  metadata.modifiedTime = projectData->modifiedTime;
  metadata.versionMajor = projectData->versionMajor;
  metadata.versionMinor = projectData->versionMinor;
  metadata.tempo = projectData->tempo;
  metadata.timeSignature = projectData->timeSignature;

  return metadata;
}

//==============================================================================
// ProjectData Serialization
//==============================================================================

juce::ValueTree ProjectManager::ProjectData::toValueTree() const {
  juce::ValueTree tree("Project");

  // Metadata
  tree.setProperty("schemaVersion", juce::String(kProjectSchemaVersion), nullptr);
  appendInvariantTree(tree, kProjectInvariants);
  tree.setProperty(
      "version", juce::String(versionMajor) + "." + juce::String(versionMinor),
      nullptr);
  tree.setProperty("name", name, nullptr);
  tree.setProperty("createdTime", createdTime.toMilliseconds(), nullptr);
  tree.setProperty("modifiedTime", modifiedTime.toMilliseconds(), nullptr);

  // Project settings
  tree.setProperty("tempo", tempo, nullptr);
  auto timeSigTree = juce::ValueTree("TimeSignature");
  timeSigTree.setProperty("numerator", timeSignature.numerator, nullptr);
  timeSigTree.setProperty("denominator", timeSignature.denominator, nullptr);
  tree.appendChild(timeSigTree, nullptr);

  // Plugin state (use full preset ValueTree serialization)
  auto pluginStateTree = pluginState.toValueTree();
  tree.appendChild(pluginStateTree, nullptr);

  // Generated MIDI (simplified - store as JSON string for now)
  // Full implementation would serialize each track separately
  tree.setProperty("hasGeneratedMidi",
                   hasOptionalItems(generatedMidi.melody) ||
                       hasOptionalItems(generatedMidi.bass) ||
                       !generatedMidi.chords.empty(),
                   nullptr);

  // Vocal notes
  auto vocalNotesTree = juce::ValueTree("VocalNotes");
  for (const auto &note : vocalNotes) {
    auto noteTree = juce::ValueTree("Note");
    noteTree.setProperty("pitch", note.pitch, nullptr);
    noteTree.setProperty("velocity", note.velocity, nullptr);
    noteTree.setProperty("startTick", static_cast<juce::int64>(note.startTick),
                         nullptr);
    noteTree.setProperty("durationTicks",
                         static_cast<juce::int64>(note.durationTicks), nullptr);
    vocalNotesTree.appendChild(noteTree, nullptr);
  }
  tree.appendChild(vocalNotesTree, nullptr);

  // Lyrics
  auto lyricsTree = juce::ValueTree("Lyrics");
  for (const auto &lyric : lyrics) {
    auto lyricTree = juce::ValueTree("Line");
    lyricTree.setProperty("text", lyric, nullptr);
    lyricsTree.appendChild(lyricTree, nullptr);
  }
  tree.appendChild(lyricsTree, nullptr);

  // Emotion selections
  auto emotionsTree = juce::ValueTree("EmotionSelections");
  for (int emotionId : selectedEmotionIds) {
    auto emotionTree = juce::ValueTree("Emotion");
    emotionTree.setProperty("id", emotionId, nullptr);
    emotionsTree.appendChild(emotionTree, nullptr);
  }
  tree.appendChild(emotionsTree, nullptr);

  if (primaryEmotionId.has_value()) {
    tree.setProperty("primaryEmotionId", *primaryEmotionId, nullptr);
  }

  return tree;
}

std::optional<ProjectManager::ProjectData>
ProjectManager::ProjectData::fromValueTree(const juce::ValueTree &tree) {
  if (!tree.isValid() || !tree.hasType("Project")) {
    return std::nullopt;
  }

  ProjectData data;
  if (tree.hasProperty("schemaVersion")) {
    data.schemaVersion = tree.getProperty("schemaVersion").toString();
  } else if (tree.hasProperty("version")) {
    data.schemaVersion = tree.getProperty("version").toString();
  }
  auto invariantsTree = tree.getChildWithName("Invariants");
  if (invariantsTree.isValid()) {
    for (int i = 0; i < invariantsTree.getNumChildren(); ++i) {
      const auto invariantTree = invariantsTree.getChild(i);
      if (invariantTree.hasType("Invariant") && invariantTree.hasProperty("name")) {
        data.invariants.push_back(invariantTree.getProperty("name").toString());
      }
    }
  }

  // Metadata
  if (tree.hasProperty("version")) {
    juce::String version = tree.getProperty("version").toString();
    int dotPos = version.indexOfChar('.');
    if (dotPos > 0) {
      data.versionMajor = version.substring(0, dotPos).getIntValue();
      data.versionMinor = version.substring(dotPos + 1).getIntValue();
    }
  }
  data.name = tree.getProperty("name").toString();
  if (tree.hasProperty("createdTime")) {
    data.createdTime = juce::Time(
        tree.getProperty("createdTime").toString().getLargeIntValue());
  }
  if (tree.hasProperty("modifiedTime")) {
    data.modifiedTime = juce::Time(
        tree.getProperty("modifiedTime").toString().getLargeIntValue());
  }

  // Project settings
  data.tempo = static_cast<float>(tree.getProperty("tempo"));
  auto timeSigTree = tree.getChildWithName("TimeSignature");
  if (timeSigTree.isValid()) {
    data.timeSignature.numerator =
        static_cast<int>(timeSigTree.getProperty("numerator"));
    data.timeSignature.denominator =
        static_cast<int>(timeSigTree.getProperty("denominator"));
  }

  // Plugin state (load full preset including CassetteState)
  auto pluginStateTree = tree.getChildWithName("Preset");
  if (pluginStateTree.isValid()) {
    auto preset = kelly::PluginState::Preset::fromValueTree(pluginStateTree);
    if (preset.has_value()) {
      // Store full preset (including CassetteState)
      data.pluginState = *preset;
    }
  }

  // Generated MIDI (restore metadata - full note restoration would be in v1.1)
  // For v1.0, we restore metadata and the user can regenerate MIDI
  if (tree.hasProperty("hasGeneratedMidi")) {
    // MIDI was saved, but we'll restore metadata only
    // User will need to regenerate MIDI (acceptable for v1.0)
  }

  // Vocal notes
  auto vocalNotesTree = tree.getChildWithName("VocalNotes");
  if (vocalNotesTree.isValid()) {
    for (int i = 0; i < vocalNotesTree.getNumChildren(); ++i) {
      auto noteTree = vocalNotesTree.getChild(i);
      if (noteTree.hasType("Note")) {
        MidiNote note;
        note.pitch = static_cast<int>(noteTree.getProperty("pitch"));
        note.velocity = static_cast<int>(noteTree.getProperty("velocity"));
        note.startTick = static_cast<int>(
            noteTree.getProperty("startTick").toString().getLargeIntValue());
        note.durationTicks =
            static_cast<int>(noteTree.getProperty("durationTicks")
                                 .toString()
                                 .getLargeIntValue());
        data.vocalNotes.push_back(note);
      }
    }
  }

  // Lyrics
  auto lyricsTree = tree.getChildWithName("Lyrics");
  if (lyricsTree.isValid()) {
    for (int i = 0; i < lyricsTree.getNumChildren(); ++i) {
      auto lyricTree = lyricsTree.getChild(i);
      if (lyricTree.hasType("Line")) {
        data.lyrics.push_back(lyricTree.getProperty("text").toString());
      }
    }
  }

  // Emotion selections
  auto emotionsTree = tree.getChildWithName("EmotionSelections");
  if (emotionsTree.isValid()) {
    for (int i = 0; i < emotionsTree.getNumChildren(); ++i) {
      auto emotionTree = emotionsTree.getChild(i);
      if (emotionTree.hasType("Emotion")) {
        int emotionId = static_cast<int>(emotionTree.getProperty("id"));
        data.selectedEmotionIds.push_back(emotionId);
      }
    }
  }

  if (tree.hasProperty("primaryEmotionId")) {
    data.primaryEmotionId =
        static_cast<int>(tree.getProperty("primaryEmotionId"));
  }

  return data;
}

ProjectManager::InvariantCheck ProjectManager::ProjectData::verifyInvariants() const {
  InvariantCheck check;

  if (schemaVersion.isEmpty()) {
    check.message = "Project artifact is corrupted: missing schemaVersion.";
    return check;
  }

  if (schemaVersion != kProjectSchemaVersion) {
    check.obsolete = true;
    check.message = "Project artifact is obsolete: schemaVersion "
        + schemaVersion + " is not supported. Expected "
        + juce::String(kProjectSchemaVersion) + ".";
    return check;
  }

  if (invariants.size() != kProjectInvariants.size()) {
    check.message = "Project artifact is corrupted: invariant list is missing or incomplete.";
    return check;
  }

  for (size_t i = 0; i < kProjectInvariants.size(); ++i) {
    if (invariants[i] != juce::String(kProjectInvariants[i])) {
      check.message = "Project artifact is corrupted: invariant contract mismatch at position "
          + juce::String(static_cast<int>(i)) + ".";
      return check;
    }
  }

  if (name.trim().isEmpty()) {
    check.message = "Project artifact is corrupted: missing project name.";
    return check;
  }

  if (tempo < 20.0f || tempo > 999.0f) {
    check.message = "Project artifact is corrupted: tempo is out of range.";
    return check;
  }

  if (timeSignature.numerator <= 0 || timeSignature.denominator <= 0) {
    check.message = "Project artifact is corrupted: invalid time signature.";
    return check;
  }

  if (pluginState.name.isEmpty()) {
    check.message = "Project artifact is corrupted: nested plugin state is missing.";
    return check;
  }

  check.ok = true;
  return check;
}

juce::var ProjectManager::ProjectData::toJson() const {
  juce::DynamicObject::Ptr obj = new juce::DynamicObject();

  // Metadata
  obj->setProperty("schemaVersion", juce::String(kProjectSchemaVersion));
  obj->setProperty("invariants", juce::var(buildInvariantArray(kProjectInvariants)));
  obj->setProperty("version", juce::String(versionMajor) + "." +
                                  juce::String(versionMinor));
  obj->setProperty("name", name);
  obj->setProperty("createdTime", createdTime.toMilliseconds());
  obj->setProperty("modifiedTime", modifiedTime.toMilliseconds());

  // Project settings
  obj->setProperty("tempo", tempo);
  juce::DynamicObject::Ptr timeSigObj = new juce::DynamicObject();
  timeSigObj->setProperty("numerator", timeSignature.numerator);
  timeSigObj->setProperty("denominator", timeSignature.denominator);
  obj->setProperty("timeSignature", juce::var(timeSigObj.get()));

  // Plugin state (serialize using existing PluginState::Preset serialization)
  obj->setProperty("pluginState", pluginState.toJson());

  // Generated MIDI (serialize full note data for all tracks)
  juce::DynamicObject::Ptr midiObj = new juce::DynamicObject();
  midiObj->setProperty("tempoBpm", generatedMidi.tempoBpm);
  midiObj->setProperty("bars", generatedMidi.bars);
  midiObj->setProperty("key", juce::String(generatedMidi.key));
  midiObj->setProperty("mode", juce::String(generatedMidi.mode));
  midiObj->setProperty("bpm", generatedMidi.bpm);
  midiObj->setProperty("lengthInBeats", generatedMidi.lengthInBeats);

  // Serialize all note tracks (create temporary ProjectManager to use instance
  // methods)
  ProjectManager tempManager;
  juce::Array<juce::var> melodyArray;
  if (generatedMidi.melody.has_value()) {
    for (const auto &note : *generatedMidi.melody) {
      melodyArray.add(tempManager.serializeMidiNote(note));
    }
  }
  midiObj->setProperty("melody", juce::var(melodyArray));

  juce::Array<juce::var> bassArray;
  if (generatedMidi.bass.has_value()) {
    for (const auto &note : *generatedMidi.bass) {
      bassArray.add(tempManager.serializeMidiNote(note));
    }
  }
  midiObj->setProperty("bass", juce::var(bassArray));

  juce::Array<juce::var> counterMelodyArray;
  if (generatedMidi.counterMelody.has_value()) {
    for (const auto &note : *generatedMidi.counterMelody) {
      counterMelodyArray.add(tempManager.serializeMidiNote(note));
    }
  }
  midiObj->setProperty("counterMelody", juce::var(counterMelodyArray));

  juce::Array<juce::var> padArray;
  if (generatedMidi.pad.has_value()) {
    for (const auto &note : *generatedMidi.pad) {
      padArray.add(tempManager.serializeMidiNote(note));
    }
  }
  midiObj->setProperty("pad", juce::var(padArray));

  juce::Array<juce::var> stringsArray;
  if (generatedMidi.strings.has_value()) {
    for (const auto &note : *generatedMidi.strings) {
      stringsArray.add(tempManager.serializeMidiNote(note));
    }
  }
  midiObj->setProperty("strings", juce::var(stringsArray));

  juce::Array<juce::var> fillsArray;
  if (generatedMidi.fills.has_value()) {
    for (const auto &note : *generatedMidi.fills) {
      fillsArray.add(tempManager.serializeMidiNote(note));
    }
  }
  midiObj->setProperty("fills", juce::var(fillsArray));

  juce::Array<juce::var> rhythmArray;
  if (generatedMidi.rhythm.has_value()) {
    for (const auto &note : *generatedMidi.rhythm) {
      rhythmArray.add(tempManager.serializeMidiNote(note));
    }
  }
  midiObj->setProperty("rhythm", juce::var(rhythmArray));

  juce::Array<juce::var> drumGrooveArray;
  if (generatedMidi.drumGroove.has_value()) {
    for (const auto &note : *generatedMidi.drumGroove) {
      drumGrooveArray.add(tempManager.serializeMidiNote(note));
    }
  }
  midiObj->setProperty("drumGroove", juce::var(drumGrooveArray));

  // Serialize chords
  juce::Array<juce::var> chordsArray;
  for (const auto &chord : generatedMidi.chords) {
    chordsArray.add(tempManager.serializeChord(chord));
  }
  midiObj->setProperty("chords", juce::var(chordsArray));

  obj->setProperty("generatedMidi", juce::var(midiObj.get()));

  // Vocal notes
  juce::Array<juce::var> vocalNotesArray;
  for (const auto &note : vocalNotes) {
    juce::DynamicObject::Ptr noteObj = new juce::DynamicObject();
    noteObj->setProperty("pitch", note.pitch);
    noteObj->setProperty("velocity", note.velocity);
    noteObj->setProperty("startTick", static_cast<juce::int64>(note.startTick));
    noteObj->setProperty("durationTicks",
                         static_cast<juce::int64>(note.durationTicks));
    vocalNotesArray.add(juce::var(noteObj.get()));
  }
  obj->setProperty("vocalNotes", juce::var(vocalNotesArray));

  // Lyrics
  juce::Array<juce::var> lyricsArray;
  for (const auto &lyric : lyrics) {
    lyricsArray.add(lyric);
  }
  obj->setProperty("lyrics", juce::var(lyricsArray));

  // Emotion selections
  juce::Array<juce::var> emotionsArray;
  for (int emotionId : selectedEmotionIds) {
    emotionsArray.add(emotionId);
  }
  obj->setProperty("selectedEmotionIds", juce::var(emotionsArray));

  if (primaryEmotionId.has_value()) {
    obj->setProperty("primaryEmotionId", *primaryEmotionId);
  }

  return juce::var(obj.get());
}

std::optional<ProjectManager::ProjectData>
ProjectManager::ProjectData::fromJson(const juce::var &json) {
  if (!json.isObject()) {
    return std::nullopt;
  }

  auto *obj = json.getDynamicObject();
  if (obj == nullptr) {
    return std::nullopt;
  }
  if (!obj->hasProperty("schemaVersion") && !obj->hasProperty("version")) {
    return std::nullopt;
  }

  ProjectData data;
  if (obj->hasProperty("schemaVersion")) {
    data.schemaVersion = obj->getProperty("schemaVersion").toString();
  } else if (obj->hasProperty("version")) {
    data.schemaVersion = obj->getProperty("version").toString();
  }
  if (obj->hasProperty("invariants")) {
    auto invariantsVar = obj->getProperty("invariants");
    if (invariantsVar.isArray()) {
      if (auto *array = invariantsVar.getArray()) {
        for (const auto &invariant : *array) {
          data.invariants.push_back(invariant.toString());
        }
      }
    }
  }

  // Metadata
  if (obj->hasProperty("version")) {
    juce::String version = obj->getProperty("version").toString();
    int dotPos = version.indexOfChar('.');
    if (dotPos > 0) {
      data.versionMajor = version.substring(0, dotPos).getIntValue();
      data.versionMinor = version.substring(dotPos + 1).getIntValue();
    }
  }
  data.name = obj->getProperty("name").toString();
  if (obj->hasProperty("createdTime")) {
    juce::int64 ms = static_cast<juce::int64>(obj->getProperty("createdTime"));
    data.createdTime = juce::Time(ms);
  }
  if (obj->hasProperty("modifiedTime")) {
    juce::int64 ms = static_cast<juce::int64>(obj->getProperty("modifiedTime"));
    data.modifiedTime = juce::Time(ms);
  }

  // Project settings
  data.tempo = static_cast<float>(obj->getProperty("tempo"));
  if (obj->hasProperty("timeSignature")) {
    auto timeSigVar = obj->getProperty("timeSignature");
    if (timeSigVar.isObject()) {
      auto *timeSigObj = timeSigVar.getDynamicObject();
      if (timeSigObj) {
        data.timeSignature.numerator =
            static_cast<int>(timeSigObj->getProperty("numerator"));
        data.timeSignature.denominator =
            static_cast<int>(timeSigObj->getProperty("denominator"));
      }
    }
  }

  // Plugin state (deserialize using existing PluginState::Preset
  // deserialization)
  if (obj->hasProperty("pluginState")) {
    auto preset =
        kelly::PluginState::Preset::fromJson(obj->getProperty("pluginState"));
    if (preset.has_value()) {
      // Store full preset (including CassetteState)
      data.pluginState = *preset;
    }
  }

  // Generated MIDI (deserialize full note data for all tracks)
  if (obj->hasProperty("generatedMidi")) {
    auto midiVar = obj->getProperty("generatedMidi");
    if (midiVar.isObject()) {
      auto *midiObj = midiVar.getDynamicObject();
      if (midiObj) {
        data.generatedMidi.tempoBpm =
            static_cast<int>(midiObj->getProperty("tempoBpm"));
        data.generatedMidi.bars =
            static_cast<int>(midiObj->getProperty("bars"));
        data.generatedMidi.key =
            midiObj->getProperty("key").toString().toStdString();
        data.generatedMidi.mode =
            midiObj->getProperty("mode").toString().toStdString();
        data.generatedMidi.bpm =
            static_cast<float>(midiObj->getProperty("bpm"));
        data.generatedMidi.lengthInBeats =
            static_cast<double>(midiObj->getProperty("lengthInBeats"));

        // Deserialize all note tracks
        ProjectManager tempManager;
        if (midiObj->hasProperty("melody")) {
          auto melodyVar = midiObj->getProperty("melody");
          if (melodyVar.isArray()) {
            auto *array = melodyVar.getArray();
            for (int i = 0; i < array->size(); ++i) {
              MidiNote note;
              if (tempManager.deserializeMidiNote(array->getReference(i),
                                                  note)) {
                ensureOptionalItems(data.generatedMidi.melody).push_back(note);
              }
            }
          }
        }

        if (midiObj->hasProperty("bass")) {
          auto bassVar = midiObj->getProperty("bass");
          if (bassVar.isArray()) {
            auto *array = bassVar.getArray();
            for (int i = 0; i < array->size(); ++i) {
              MidiNote note;
              if (tempManager.deserializeMidiNote(array->getReference(i),
                                                  note)) {
                ensureOptionalItems(data.generatedMidi.bass).push_back(note);
              }
            }
          }
        }

        if (midiObj->hasProperty("counterMelody")) {
          auto counterMelodyVar = midiObj->getProperty("counterMelody");
          if (counterMelodyVar.isArray()) {
            auto *array = counterMelodyVar.getArray();
            for (int i = 0; i < array->size(); ++i) {
              MidiNote note;
              if (tempManager.deserializeMidiNote(array->getReference(i),
                                                  note)) {
                ensureOptionalItems(data.generatedMidi.counterMelody).push_back(note);
              }
            }
          }
        }

        if (midiObj->hasProperty("pad")) {
          auto padVar = midiObj->getProperty("pad");
          if (padVar.isArray()) {
            auto *array = padVar.getArray();
            for (int i = 0; i < array->size(); ++i) {
              MidiNote note;
              if (tempManager.deserializeMidiNote(array->getReference(i),
                                                  note)) {
                ensureOptionalItems(data.generatedMidi.pad).push_back(note);
              }
            }
          }
        }

        if (midiObj->hasProperty("strings")) {
          auto stringsVar = midiObj->getProperty("strings");
          if (stringsVar.isArray()) {
            auto *array = stringsVar.getArray();
            for (int i = 0; i < array->size(); ++i) {
              MidiNote note;
              if (tempManager.deserializeMidiNote(array->getReference(i),
                                                  note)) {
                ensureOptionalItems(data.generatedMidi.strings).push_back(note);
              }
            }
          }
        }

        if (midiObj->hasProperty("fills")) {
          auto fillsVar = midiObj->getProperty("fills");
          if (fillsVar.isArray()) {
            auto *array = fillsVar.getArray();
            for (int i = 0; i < array->size(); ++i) {
              MidiNote note;
              if (tempManager.deserializeMidiNote(array->getReference(i),
                                                  note)) {
                ensureOptionalItems(data.generatedMidi.fills).push_back(note);
              }
            }
          }
        }

        if (midiObj->hasProperty("rhythm")) {
          auto rhythmVar = midiObj->getProperty("rhythm");
          if (rhythmVar.isArray()) {
            auto *array = rhythmVar.getArray();
            for (int i = 0; i < array->size(); ++i) {
              MidiNote note;
              if (tempManager.deserializeMidiNote(array->getReference(i),
                                                  note)) {
                ensureOptionalItems(data.generatedMidi.rhythm).push_back(note);
              }
            }
          }
        }

        if (midiObj->hasProperty("drumGroove")) {
          auto drumGrooveVar = midiObj->getProperty("drumGroove");
          if (drumGrooveVar.isArray()) {
            auto *array = drumGrooveVar.getArray();
            for (int i = 0; i < array->size(); ++i) {
              MidiNote note;
              if (tempManager.deserializeMidiNote(array->getReference(i),
                                                  note)) {
                ensureOptionalItems(data.generatedMidi.drumGroove).push_back(note);
              }
            }
          }
        }

        // Deserialize chords
        if (midiObj->hasProperty("chords")) {
          auto chordsVar = midiObj->getProperty("chords");
          if (chordsVar.isArray()) {
            auto *array = chordsVar.getArray();
            for (int i = 0; i < array->size(); ++i) {
              Chord chord;
              if (tempManager.deserializeChord(array->getReference(i), chord)) {
                data.generatedMidi.chords.push_back(chord);
              }
            }
          }
        }
      }
    }
  }

  // Vocal notes
  if (obj->hasProperty("vocalNotes")) {
    auto vocalNotesVar = obj->getProperty("vocalNotes");
    if (vocalNotesVar.isArray()) {
      auto *array = vocalNotesVar.getArray();
      for (int i = 0; i < array->size(); ++i) {
        auto noteVar = array->getReference(i);
        if (noteVar.isObject()) {
          auto *noteObj = noteVar.getDynamicObject();
          if (noteObj) {
            MidiNote note;
            note.pitch = static_cast<int>(noteObj->getProperty("pitch"));
            note.velocity = static_cast<int>(noteObj->getProperty("velocity"));
            note.startTick = static_cast<int>(noteObj->getProperty("startTick")
                                                  .toString()
                                                  .getLargeIntValue());
            note.durationTicks =
                static_cast<int>(noteObj->getProperty("durationTicks")
                                     .toString()
                                     .getLargeIntValue());
            data.vocalNotes.push_back(note);
          }
        }
      }
    }
  }

  // Lyrics
  if (obj->hasProperty("lyrics")) {
    auto lyricsVar = obj->getProperty("lyrics");
    if (lyricsVar.isArray()) {
      auto *array = lyricsVar.getArray();
      for (int i = 0; i < array->size(); ++i) {
        data.lyrics.push_back(array->getReference(i).toString());
      }
    }
  }

  // Emotion selections
  if (obj->hasProperty("selectedEmotionIds")) {
    auto emotionsVar = obj->getProperty("selectedEmotionIds");
    if (emotionsVar.isArray()) {
      auto *array = emotionsVar.getArray();
      for (int i = 0; i < array->size(); ++i) {
        data.selectedEmotionIds.push_back(
            static_cast<int>(array->getReference(i)));
      }
    }
  }

  if (obj->hasProperty("primaryEmotionId")) {
    data.primaryEmotionId =
        static_cast<int>(obj->getProperty("primaryEmotionId"));
  }

  return data;
}

//==============================================================================
// Helper Methods
//==============================================================================

juce::var
ProjectManager::serializeGeneratedMidi(const GeneratedMidi &midi) const {
  juce::DynamicObject::Ptr obj = new juce::DynamicObject();

  obj->setProperty("tempoBpm", midi.tempoBpm);
  obj->setProperty("bars", midi.bars);
  obj->setProperty("key", juce::String(midi.key));
  obj->setProperty("mode", juce::String(midi.mode));
  obj->setProperty("bpm", midi.bpm);
  obj->setProperty("lengthInBeats", midi.lengthInBeats);

  // Serialize tracks (simplified - store note counts for now)
  // Full implementation would serialize all notes
  obj->setProperty("melodyNoteCount",
                   static_cast<int>(midi.melody.has_value() ? midi.melody->size() : 0));
  obj->setProperty("bassNoteCount",
                   static_cast<int>(midi.bass.has_value() ? midi.bass->size() : 0));
  obj->setProperty("chordCount", static_cast<int>(midi.chords.size()));

  return juce::var(obj.get());
}

bool ProjectManager::deserializeGeneratedMidi(const juce::var &json,
                                              GeneratedMidi &outMidi) const {
  if (!json.isObject()) {
    return false;
  }

  auto *obj = json.getDynamicObject();
  if (!obj) {
    return false;
  }

  outMidi.tempoBpm = static_cast<int>(obj->getProperty("tempoBpm"));
  outMidi.bars = static_cast<int>(obj->getProperty("bars"));
  outMidi.key = obj->getProperty("key").toString().toStdString();
  outMidi.mode = obj->getProperty("mode").toString().toStdString();
  outMidi.bpm = static_cast<float>(obj->getProperty("bpm"));
  outMidi.lengthInBeats =
      static_cast<double>(obj->getProperty("lengthInBeats"));

  // Note: Full deserialization would restore all notes
  // For now, we just restore metadata
  // In production, you'd want to serialize/deserialize all notes

  return true;
}

juce::var ProjectManager::serializeMidiNote(const MidiNote &note) const {
  juce::DynamicObject::Ptr obj = new juce::DynamicObject();
  obj->setProperty("pitch", note.pitch);
  obj->setProperty("velocity", note.velocity);
  obj->setProperty("startTick", static_cast<juce::int64>(note.startTick));
  obj->setProperty("durationTicks",
                   static_cast<juce::int64>(note.durationTicks));
  return juce::var(obj.get());
}

bool ProjectManager::deserializeMidiNote(const juce::var &json,
                                         MidiNote &outNote) const {
  if (!json.isObject()) {
    return false;
  }

  auto *obj = json.getDynamicObject();
  if (!obj) {
    return false;
  }

  outNote.pitch = static_cast<int>(obj->getProperty("pitch"));
  outNote.velocity = static_cast<int>(obj->getProperty("velocity"));
  outNote.startTick = static_cast<int>(
      obj->getProperty("startTick").toString().getLargeIntValue());
  outNote.durationTicks = static_cast<int>(
      obj->getProperty("durationTicks").toString().getLargeIntValue());

  return true;
}

juce::var ProjectManager::serializeChord(const Chord &chord) const {
  juce::DynamicObject::Ptr obj = new juce::DynamicObject();
  obj->setProperty("symbol", juce::String(chord.symbol));
  obj->setProperty("root", juce::String(chord.root));
  obj->setProperty("quality", juce::String(chord.quality));
  obj->setProperty("startBeat", chord.startBeat);
  obj->setProperty("duration", chord.duration);

  juce::Array<juce::var> pitchesArray;
  for (int pitch : chord.pitches) {
    pitchesArray.add(pitch);
  }
  obj->setProperty("pitches", juce::var(pitchesArray));

  return juce::var(obj.get());
}

bool ProjectManager::deserializeChord(const juce::var &json,
                                      Chord &outChord) const {
  if (!json.isObject()) {
    return false;
  }

  auto *obj = json.getDynamicObject();
  if (!obj) {
    return false;
  }

  outChord.symbol = obj->getProperty("symbol").toString().toStdString();
  outChord.root = obj->getProperty("root").toString().toStdString();
  outChord.quality = obj->getProperty("quality").toString().toStdString();
  outChord.startBeat = static_cast<double>(obj->getProperty("startBeat"));
  outChord.duration = static_cast<double>(obj->getProperty("duration"));

  if (obj->hasProperty("pitches")) {
    auto pitchesVar = obj->getProperty("pitches");
    if (pitchesVar.isArray()) {
      auto *array = pitchesVar.getArray();
      for (int i = 0; i < array->size(); ++i) {
        outChord.pitches.push_back(static_cast<int>(array->getReference(i)));
      }
    }
  }

  return true;
}

bool ProjectManager::migrateProjectVersion(juce::ValueTree &tree,
                                           int fromVersion,
                                           int toVersion) const {
  // Version migration logic
  // For now, version 1.0 is the only version
  if (fromVersion == 1 && toVersion == 1) {
    return true; // No migration needed
  }

  // Future: Add migration logic for version 1.0 -> 1.1, etc.
  return false;
}

} // namespace midikompanion
