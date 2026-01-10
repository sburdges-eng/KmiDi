#include "bridge/BridgeBase.h"
#include <iostream>

namespace kelly {
namespace bridge {

BridgeBase::BridgeBase(const std::string& bridgeName)
    : bridgeName_(bridgeName)
    , available_(false)
{
}

void BridgeBase::logError(const std::string& message) const {
    std::cerr << "[" << bridgeName_ << "] ERROR: " << message << std::endl;
}

void BridgeBase::logInfo(const std::string& message) const {
    std::cout << "[" << bridgeName_ << "] " << message << std::endl;
}

} // namespace bridge
} // namespace kelly
