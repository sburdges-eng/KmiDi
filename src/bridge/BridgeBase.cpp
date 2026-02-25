#include "bridge/BridgeBase.h"
#include <iostream>

namespace kelly {
namespace bridge {

BridgeBase::BridgeBase(const std::string& bridgeName)
    : available_(false)
    , bridgeName_(bridgeName)
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
