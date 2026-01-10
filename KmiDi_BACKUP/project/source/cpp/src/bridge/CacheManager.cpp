#include "bridge/CacheManager.h"
#include <algorithm>

namespace kelly {
namespace bridge {

CacheManager::CacheManager(int ttlMs, size_t maxSize)
    : ttlMs_(ttlMs)
    , maxSize_(maxSize)
{
}

std::string CacheManager::get(const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = cache_.find(key);
    if (it == cache_.end()) {
        return "";
    }

    // Check if expired
    auto now = std::chrono::steady_clock::now();
    auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - it->second.timestamp
    ).count();

    if (age > ttlMs_) {
        cache_.erase(it);
        return "";
    }

    return it->second.value;
}

void CacheManager::put(const std::string& key, const std::string& value) {
    std::lock_guard<std::mutex> lock(mutex_);

    CacheEntry entry;
    entry.value = value;
    entry.timestamp = std::chrono::steady_clock::now();

    cache_[key] = entry;

    // Enforce size limit
    enforceSizeLimit();
}

bool CacheManager::exists(const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = cache_.find(key);
    if (it == cache_.end()) {
        return false;
    }

    // Check if expired
    auto now = std::chrono::steady_clock::now();
    auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - it->second.timestamp
    ).count();

    if (age > ttlMs_) {
        cache_.erase(it);
        return false;
    }

    return true;
}

void CacheManager::remove(const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.erase(key);
}

void CacheManager::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
}

void CacheManager::pruneExpired() {
    std::lock_guard<std::mutex> lock(mutex_);

    auto now = std::chrono::steady_clock::now();
    auto it = cache_.begin();

    while (it != cache_.end()) {
        auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - it->second.timestamp
        ).count();

        if (age > ttlMs_) {
            it = cache_.erase(it);
        } else {
            ++it;
        }
    }
}

size_t CacheManager::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cache_.size();
}

void CacheManager::enforceSizeLimit() {
    if (cache_.size() <= maxSize_) {
        return;
    }

    // Remove oldest entries (simple: remove first 20% over limit)
    size_t toRemove = (cache_.size() - maxSize_) + (maxSize_ / 5);
    
    // Find oldest entries
    std::vector<std::map<std::string, CacheEntry>::iterator> sorted;
    for (auto it = cache_.begin(); it != cache_.end(); ++it) {
        sorted.push_back(it);
    }

    std::sort(sorted.begin(), sorted.end(),
        [](const auto& a, const auto& b) {
            return a->second.timestamp < b->second.timestamp;
        });

    // Remove oldest
    for (size_t i = 0; i < toRemove && i < sorted.size(); ++i) {
        cache_.erase(sorted[i]);
    }
}

} // namespace bridge
} // namespace kelly
