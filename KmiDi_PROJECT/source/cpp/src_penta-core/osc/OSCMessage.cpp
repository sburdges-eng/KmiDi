#include "penta/osc/OSCMessage.h"
#include <stdexcept>

namespace penta {
namespace osc {

OSCMessage::OSCMessage(const std::string& addressPattern)
    : addressPattern_(addressPattern)
    , arguments_()
    , timestamp_(0)
{}

void OSCMessage::addInt(int32_t value) {
    arguments_.emplace_back(value);
}

void OSCMessage::addFloat(float value) {
    arguments_.emplace_back(value);
}

void OSCMessage::addString(const std::string& value) {
    arguments_.emplace_back(value);
}

void OSCMessage::addBlob(const std::vector<uint8_t>& value) {
    arguments_.emplace_back(value);
}

const OSCArgument& OSCMessage::getArgument(size_t index) const {
    if (index >= arguments_.size()) {
        throw std::out_of_range("OSCMessage::getArgument: Index out of range");
    }
    return arguments_[index];
}

int32_t OSCMessage::getInt(size_t index, int32_t defaultValue) const {
    if (index < arguments_.size()) {
        if (std::holds_alternative<int32_t>(arguments_[index])) {
            return std::get<int32_t>(arguments_[index]);
        }
    }
    return defaultValue;
}

float OSCMessage::getFloat(size_t index, float defaultValue) const {
    if (index < arguments_.size()) {
        if (std::holds_alternative<float>(arguments_[index])) {
            return std::get<float>(arguments_[index]);
        }
    }
    return defaultValue;
}

std::string OSCMessage::getString(size_t index, const std::string& defaultValue) const {
    if (index < arguments_.size()) {
        if (std::holds_alternative<std::string>(arguments_[index])) {
            return std::get<std::string>(arguments_[index]);
        }
    }
    return defaultValue;
}

std::vector<uint8_t> OSCMessage::getBlob(size_t index, const std::vector<uint8_t>& defaultValue) const {
    if (index < arguments_.size()) {
        if (std::holds_alternative<std::vector<uint8_t>>(arguments_[index])) {
            return std::get<std::vector<uint8_t>>(arguments_[index]);
        }
    }
    return defaultValue;
}

} // namespace osc
} // namespace penta
