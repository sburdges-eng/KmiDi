#include "plugin/PluginIRInspector.h"
#include <iomanip>
#include <sstream>

namespace kelly {

PluginIRInspector::PluginIRInspector() {
  titleLabel_.setText("Intent IR Inspector", juce::dontSendNotification);
  titleLabel_.setFont(juce::Font(juce::FontOptions(14.0f, juce::Font::bold)));
  addAndMakeVisible(titleLabel_);

  frameDisplay_.setMultiLine(true);
  frameDisplay_.setReadOnly(true);
  frameDisplay_.setFont(juce::Font(juce::FontOptions(
      juce::Font::getDefaultMonospacedFontName(), 11.0f, juce::Font::plain)));
  frameDisplay_.setColour(juce::TextEditor::backgroundColourId,
                          juce::Colour(0xff1e1e1e));
  frameDisplay_.setColour(juce::TextEditor::textColourId,
                          juce::Colour(0xffd4d4d4));
  addAndMakeVisible(frameDisplay_);
}

void PluginIRInspector::paint(juce::Graphics &g) {
  g.fillAll(juce::Colour(0xff252526));
  g.setColour(juce::Colour(0xff3e3e3e));
  g.drawRect(getLocalBounds(), 1);
}

void PluginIRInspector::resized() {
  auto bounds = getLocalBounds();
  titleLabel_.setBounds(bounds.removeFromTop(30));
  frameDisplay_.setBounds(bounds.reduced(4));
}

void PluginIRInspector::updateFrame(const intent_ir::IntentFrame &frame) {
  {
    std::lock_guard<std::mutex> lock(frameMutex_);
    currentFrame_ = frame;
    hasFrame_.store(true);
  }

  // Update display on message thread without capturing a raw component pointer
  juce::Component::SafePointer<PluginIRInspector> safeThis(this);
  juce::MessageManager::callAsync([safeThis]() {
    if (safeThis != nullptr) {
      safeThis->updateDisplay();
    }
  });
}

void PluginIRInspector::clear() {
  {
    std::lock_guard<std::mutex> lock(frameMutex_);
    hasFrame_.store(false);
  }
  frameDisplay_.clear();
}

void PluginIRInspector::updateDisplay() {
  if (!hasFrame_.load()) {
    frameDisplay_.setText("No IntentFrame available",
                          juce::dontSendNotification);
    return;
  }

  intent_ir::IntentFrame frame;
  {
    std::lock_guard<std::mutex> lock(frameMutex_);
    frame = currentFrame_;
  }

  frameDisplay_.setText(formatFrame(frame), juce::dontSendNotification);
}

juce::String
PluginIRInspector::formatFrame(const intent_ir::IntentFrame &frame) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(3);

  oss << "Intent IR v" << frame.meta.ir_version << "\n";
  oss << "Intent ID: 0x" << std::hex << frame.meta.intent_id << std::dec
      << "\n";
  oss << "Session ID: 0x" << std::hex << frame.meta.session_id << std::dec
      << "\n\n";

  oss << "Emotion:\n";
  oss << "  Valence: " << frame.emotion.valence << "\n";
  oss << "  Arousal: " << frame.emotion.arousal << "\n";
  oss << "  Dominance: " << frame.emotion.dominance << "\n";
  oss << "  Discrete ID: " << frame.emotion.discrete_id << "\n";
  oss << "  Intensity: " << frame.emotion.intensity << "\n";
  oss << "  Confidence: " << frame.emotion.confidence << "\n\n";

  oss << "Music:\n";
  oss << "  Tempo Bias: " << frame.music.tempo_bias << "\n";
  oss << "  Rhythmic Density: " << frame.music.rhythmic_density << "\n";
  oss << "  Groove Strength: " << frame.music.groove_strength << "\n";
  oss << "  Harmonic Tension: " << frame.music.harmonic_tension << "\n";
  oss << "  Harmonic Motion: " << frame.music.harmonic_motion << "\n";
  oss << "  Mode Preference: " << static_cast<int>(frame.music.mode_preference)
      << "\n";
  oss << "  Melodic Activity: " << frame.music.melodic_activity << "\n";
  oss << "  Contour Variance: " << frame.music.contour_variance << "\n";
  oss << "  Dynamic Range: " << frame.music.dynamic_range << "\n";
  oss << "  Texture Density: " << frame.music.texture_density << "\n\n";

  oss << "Time:\n";
  oss << "  Start Bar: " << frame.time.start_bar << "\n";
  oss << "  End Bar: " << frame.time.end_bar << "\n";
  oss << "  Fade In: " << frame.time.fade_in_beats << " beats\n";
  oss << "  Fade Out: " << frame.time.fade_out_beats << " beats\n\n";

  oss << "Provenance:\n";
  oss << "  Source: "
      << getProvenanceName(frame.provenance.source).toStdString() << "\n";
  oss << "  User Override Weight: " << frame.provenance.user_override_weight
      << "\n";

  return juce::String(oss.str());
}

juce::Colour
PluginIRInspector::getProvenanceColor(intent_ir::IntentSource source) {
  switch (source) {
  case intent_ir::IntentSource::UiDirect:
    return juce::Colour(0xff4CAF50); // Green
  case intent_ir::IntentSource::UiEdit:
    return juce::Colour(0xff2196F3); // Blue
  case intent_ir::IntentSource::MlText:
    return juce::Colour(0xffFF9800); // Orange
  case intent_ir::IntentSource::MlAudio:
    return juce::Colour(0xffF44336); // Red
  case intent_ir::IntentSource::Preset:
    return juce::Colour(0xff9C27B0); // Purple
  case intent_ir::IntentSource::Automation:
    return juce::Colour(0xff607D8B); // Blue-grey
  default:
    return juce::Colours::white;
  }
}

juce::String
PluginIRInspector::getProvenanceName(intent_ir::IntentSource source) {
  switch (source) {
  case intent_ir::IntentSource::UiDirect:
    return "UI_DIRECT";
  case intent_ir::IntentSource::UiEdit:
    return "UI_EDIT";
  case intent_ir::IntentSource::MlText:
    return "ML_TEXT";
  case intent_ir::IntentSource::MlAudio:
    return "ML_AUDIO";
  case intent_ir::IntentSource::Preset:
    return "PRESET";
  case intent_ir::IntentSource::Automation:
    return "AUTOMATION";
  default:
    return "UNKNOWN";
  }
}

} // namespace kelly
