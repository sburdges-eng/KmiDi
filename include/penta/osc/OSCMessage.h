#pragma once

#include <string>
#include <vector>
#include <variant>
#include <cstdint>

namespace penta {
namespace osc {

// Argument types that can be sent via OSC
using OSCArgument = std::variant<int32_t, float, std::string, std::vector<uint8_t>>; // blob

/**
 * @brief Represents an OSC message with an address pattern and arguments.
 */
class OSCMessage {
public:
    OSCMessage() = default;
    explicit OSCMessage(const std::string& addressPattern);

    const std::string& getAddress() const { return addressPattern_; }
    void setAddress(const std::string& address) { addressPattern_ = address; }

    // Add arguments
    void addInt(int32_t value);
    void addFloat(float value);
    void addString(const std::string& value);
    void addBlob(const std::vector<uint8_t>& value);

    // Get arguments
    size_t getArgumentCount() const { return arguments_.size(); }
    const OSCArgument& getArgument(size_t index) const;

    // Getters for specific types (with bounds checking)
    int32_t getInt(size_t index, int32_t defaultValue = 0) const;
    float getFloat(size_t index, float defaultValue = 0.0f) const;
    std::string getString(size_t index, const std::string& defaultValue = "") const;
    std::vector<uint8_t> getBlob(size_t index, const std::vector<uint8_t>& defaultValue = {}) const;

    // Timestamp (optional, for bundles)
    void setTimestamp(uint64_t timestamp) { timestamp_ = timestamp; }
    uint64_t getTimestamp() const { return timestamp_; }

private:
    std::string addressPattern_;
    std::vector<OSCArgument> arguments_;
    uint64_t timestamp_ = 0;
};

} // namespace osc
} // namespace penta
