#pragma once

#include <juce_audio_devices/juce_audio_devices.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <functional>
#include <vector>
#include <memory>

namespace daiw {
namespace midi {

struct MidiDeviceInfo {
    int id;
    std::string name;
    std::string identifier;
};

class MidiInput {
public:
    MidiInput();
    ~MidiInput();

    static std::vector<MidiDeviceInfo> getAvailableDevices();
    bool open(int deviceId);
    void close();
    void start();
    void stop();

    void setCallback(std::function<void(const juce::MidiMessage&)> callback);

private:
    std::unique_ptr<juce::MidiInput> juceMidiInput_;
    std::function<void(const juce::MidiMessage&)> callback_;
    int deviceId_ = -1;
    bool isOpen_ = false;
    bool isRunning_ = false;

    class CallbackWrapper : public juce::MidiInputCallback {
    public:
        CallbackWrapper(MidiInput* owner) : owner_(owner) {}
        void handleIncomingMidiMessage(juce::MidiInput* source,
                                      const juce::MidiMessage& message) override {
            if (owner_ && owner_->callback_) {
                owner_->callback_(message);
            }
        }
    private:
        MidiInput* owner_;
    };
    std::unique_ptr<CallbackWrapper> callbackWrapper_;
};

class MidiOutput {
public:
    MidiOutput();
    ~MidiOutput();

    static std::vector<MidiDeviceInfo> getAvailableDevices();
    bool open(int deviceId);
    void close();
    bool sendMessage(const juce::MidiMessage& message);

private:
    std::unique_ptr<juce::MidiOutput> juceMidiOutput_;
    int deviceId_ = -1;
    bool isOpen_ = false;
};

} // namespace midi
} // namespace daiw
