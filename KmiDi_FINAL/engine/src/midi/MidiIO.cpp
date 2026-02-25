#include "midi/MidiIO.h"
#include <juce_core/juce_core.h>

namespace daiw {
namespace midi {

MidiInput::MidiInput()
    : deviceId_(-1)
    , isOpen_(false)
    , isRunning_(false)
    , callbackWrapper_(std::make_unique<CallbackWrapper>(this))
{}

MidiInput::~MidiInput() {
    close();
}

std::vector<MidiDeviceInfo> MidiInput::getAvailableDevices() {
    std::vector<MidiDeviceInfo> devices;

    auto juceDevices = juce::MidiInput::getAvailableDevices();
    for (int i = 0; i < juceDevices.size(); ++i) {
        MidiDeviceInfo info;
        info.id = i;
        info.name = juceDevices[i].name.toStdString();
        info.identifier = juceDevices[i].identifier.toStdString();
        devices.push_back(info);
    }

    return devices;
}

bool MidiInput::open(int deviceId) {
    if (isOpen_) {
        close();
    }

    auto devices = juce::MidiInput::getAvailableDevices();

    if (deviceId < 0 || deviceId >= devices.size()) {
        juce::Logger::writeToLog("MidiInput::open: Invalid device ID: " +
                                 juce::String(deviceId));
        return false;
    }

    auto device = devices[deviceId];

    juceMidiInput_ = juce::MidiInput::openDevice(
        device.identifier,
        callbackWrapper_.get()
    );

    if (juceMidiInput_) {
        deviceId_ = deviceId;
        isOpen_ = true;
        juce::Logger::writeToLog("MidiInput::open: Successfully opened device: " +
                                 device.name);
        return true;
    }

    juce::Logger::writeToLog("MidiInput::open: Failed to open device: " +
                             device.name);
    return false;
}

void MidiInput::close() {
    stop();
    juceMidiInput_.reset();
    isOpen_ = false;
    deviceId_ = -1;
}

void MidiInput::start() {
    if (!isOpen_ || !juceMidiInput_) {
        return;
    }

    juceMidiInput_->start();
    isRunning_ = true;
}

void MidiInput::stop() {
    if (juceMidiInput_ && isRunning_) {
        juceMidiInput_->stop();
        isRunning_ = false;
    }
}

void MidiInput::setCallback(std::function<void(const juce::MidiMessage&)> callback) {
    callback_ = std::move(callback);
}

MidiOutput::MidiOutput()
    : deviceId_(-1)
    , isOpen_(false)
{}

MidiOutput::~MidiOutput() {
    close();
}

std::vector<MidiDeviceInfo> MidiOutput::getAvailableDevices() {
    std::vector<MidiDeviceInfo> devices;

    auto juceDevices = juce::MidiOutput::getAvailableDevices();
    for (int i = 0; i < juceDevices.size(); ++i) {
        MidiDeviceInfo info;
        info.id = i;
        info.name = juceDevices[i].name.toStdString();
        info.identifier = juceDevices[i].identifier.toStdString();
        devices.push_back(info);
    }

    return devices;
}

bool MidiOutput::open(int deviceId) {
    if (isOpen_) {
        close();
    }

    auto devices = juce::MidiOutput::getAvailableDevices();

    if (deviceId < 0 || deviceId >= devices.size()) {
        juce::Logger::writeToLog("MidiOutput::open: Invalid device ID: " +
                                 juce::String(deviceId));
        return false;
    }

    auto device = devices[deviceId];
    juceMidiOutput_ = juce::MidiOutput::openDevice(device.identifier);

    if (juceMidiOutput_) {
        deviceId_ = deviceId;
        isOpen_ = true;
        juce::Logger::writeToLog("MidiOutput::open: Successfully opened device: " +
                                 device.name);
        return true;
    }

    juce::Logger::writeToLog("MidiOutput::open: Failed to open device: " +
                             device.name);
    return false;
}

void MidiOutput::close() {
    juceMidiOutput_.reset();
    isOpen_ = false;
    deviceId_ = -1;
}

bool MidiOutput::sendMessage(const juce::MidiMessage& message) {
    if (!isOpen_ || !juceMidiOutput_) {
        return false;
    }

    juceMidiOutput_->sendMessageNow(message);
    return true;
}

} // namespace midi
} // namespace daiw
