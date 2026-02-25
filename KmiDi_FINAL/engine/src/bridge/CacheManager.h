#pragma once

/**
 * CacheManager.h - Generic caching utility for bridges
 * ====================================================
 *
 * Provides reusable caching logic for bridges that need TTL-based caching:
 * - TTL-based expiration
 * - Size limits
 * - Thread-safe operations
 * - Generic key-value storage
 */

#include <string>
#include <map>
#include <mutex>
#include <chrono>

namespace kelly {
namespace bridge {

/**
 * CacheManager - Generic cache with TTL and size limits
 *
 * Thread-safe cache implementation for bridge results.
 */
class CacheManager {
public:
    struct CacheEntry {
        std::string value;
        std::chrono::steady_clock::time_point timestamp;
    };

    /**
     * Constructor
     * @param ttlMs Time-to-live in milliseconds (default: 2000ms)
     * @param maxSize Maximum number of entries (default: 100)
     */
    explicit CacheManager(int ttlMs = 2000, size_t maxSize = 100);

    /**
     * Get cached value for a key
     * @param key Cache key
     * @return Cached value, or empty string if not found or expired
     */
    std::string get(const std::string& key);

    /**
     * Store a value in cache
     * @param key Cache key
     * @param value Value to cache
     */
    void put(const std::string& key, const std::string& value);

    /**
     * Check if a key exists and is valid
     * @param key Cache key
     * @return true if key exists and is not expired
     */
    bool exists(const std::string& key);

    /**
     * Remove a specific key from cache
     * @param key Cache key to remove
     */
    void remove(const std::string& key);

    /**
     * Clear all cache entries
     */
    void clear();

    /**
     * Remove expired entries (called automatically, but can be called manually)
     */
    void pruneExpired();

    /**
     * Get current cache size
     */
    size_t size() const;

    /**
     * Set TTL for new entries
     */
    void setTTL(int ttlMs) {
        std::lock_guard<std::mutex> lock(mutex_);
        ttlMs_ = ttlMs;
    }

    /**
     * Set maximum cache size
     */
    void setMaxSize(size_t maxSize) {
        std::lock_guard<std::mutex> lock(mutex_);
        maxSize_ = maxSize;
    }

private:
    std::map<std::string, CacheEntry> cache_;
    mutable std::mutex mutex_;
    int ttlMs_;
    size_t maxSize_;

    void enforceSizeLimit();
};

} // namespace bridge
} // namespace kelly
