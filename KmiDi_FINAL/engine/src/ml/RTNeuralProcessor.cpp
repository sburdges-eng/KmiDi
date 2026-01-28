#include "ml/RTNeuralProcessor.h"
#include <cstring>
#include <algorithm>
#include <fstream>
#include <juce_core/juce_core.h>
#include <juce_data_structures/juce_data_structures.h>
#ifdef ENABLE_RTNEURAL
#include <RTNeural/RTNeural.h>
#endif

namespace kelly {

bool RTNeuralProcessor::loadModel(const juce::File &jsonFile) {
    if (!jsonFile.existsAsFile()) {
        juce::Logger::writeToLog("ML Model file not found: " + jsonFile.getFullPathName());
        return false;
    }

#ifdef ENABLE_RTNEURAL
    try {
        const std::string filePath = jsonFile.getFullPathName().toStdString();
        std::ifstream jsonStream(filePath, std::ifstream::binary);
        if (!jsonStream.is_open()) {
            juce::Logger::writeToLog("Failed to open ML model file: " + jsonFile.getFullPathName());
            modelLoaded_ = false;
            return false;
        }

        auto model = RTNeural::json_parser::parseJson<float>(jsonStream);
        if (!model) {
            juce::Logger::writeToLog("Failed to parse RTNeural model JSON: " + jsonFile.getFullPathName());
            modelLoaded_ = false;
            return false;
        }

        model->reset();
        model_ = std::move(model);
        modelLoaded_ = true;
        modelPath_ = filePath;

        juce::Logger::writeToLog("ML Model loaded successfully: " + jsonFile.getFullPathName());
        return true;
    } catch (const std::exception& e) {
        juce::Logger::writeToLog("Exception while loading RTNeural model: " + juce::String(e.what()));
        modelLoaded_ = false;
        model_.reset();
        return false;
    } catch (...) {
        juce::Logger::writeToLog("Unknown exception while loading RTNeural model");
        modelLoaded_ = false;
        model_.reset();
        return false;
    }
#else
    // Basic validation to ensure JSON is readable even without RTNeural
    juce::FileInputStream stream(jsonFile);
    if (!stream.openedOk()) {
        juce::Logger::writeToLog("Failed to open ML model file: " + jsonFile.getFullPathName());
        return false;
    }

    auto parsed = juce::JSON::parse(stream.readEntireStreamAsString());
    if (parsed.isVoid()) {
        juce::Logger::writeToLog("Invalid ML model JSON: " + jsonFile.getFullPathName());
        return false;
    }

    model_.loaded = true;
    model_.modelPath = jsonFile.getFullPathName().toStdString();
    juce::Logger::writeToLog("ML Model loaded (RTNeural disabled): " + jsonFile.getFullPathName());
    return true;
#endif
}

void RTNeuralProcessor::process(const float* input, float* output, int numSamples)
{
#ifdef ENABLE_RTNEURAL
    if (!isModelLoaded() || !model_) {
        std::memcpy(output, input, static_cast<size_t>(numSamples) * sizeof(float));
        return;
    }

    try {
        model_->reset();
        for (int i = 0; i < numSamples; ++i) {
            const float sample = input[i];
            model_->forward(&sample);
            const float* out = model_->getOutputs();
            output[i] = out ? out[0] : sample;
        }
        return;
    } catch (const std::exception& e) {
        juce::Logger::writeToLog("RTNeuralProcessor::process exception: " + juce::String(e.what()));
    } catch (...) {
        juce::Logger::writeToLog("RTNeuralProcessor::process unknown exception");
    }
#endif

    // Fallback passthrough
    std::memcpy(output, input, static_cast<size_t>(numSamples) * sizeof(float));
}

std::array<float, 64>
RTNeuralProcessor::inferEmotion(const std::array<float, 128> &features) {
  std::array<float, 64> result{};

  if (!isModelLoaded()) {
    // Return simple heuristic if no model loaded
    for (size_t i = 0; i < 32; ++i) {
      result[i] = features[i] * 0.3f;           // Valence proxy
      result[i + 32] = features[i + 32] * 0.5f; // Arousal proxy
    }
    return result;
  }

#ifdef ENABLE_RTNEURAL
  if (model_) {
    try {
      model_->reset();
      model_->forward(features.data());
      if (const float* outputs = model_->getOutputs()) {
        std::copy_n(outputs, std::min<size_t>(64, result.size()), result.begin());
        return result;
      }
    } catch (const std::exception &e) {
      juce::Logger::writeToLog("Exception during RTNeural inference: " +
                               juce::String(e.what()));
    } catch (...) {
      juce::Logger::writeToLog("Unknown exception during RTNeural inference");
    }
  } else {
    juce::Logger::writeToLog("Warning: Model pointer is null in inferEmotion()");
  }
#endif

  // Fallback heuristic
  for (size_t i = 0; i < 32; ++i) {
    result[i] = features[i] * 0.3f;
    result[i + 32] = features[i + 32] * 0.5f;
  }
  return result;
}

} // namespace kelly
