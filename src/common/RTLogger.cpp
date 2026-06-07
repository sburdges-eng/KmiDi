#include "penta/common/RTLogger.h"
#include <cstring>
#include <iostream>
#include <mutex>
#if defined(__APPLE__)
#include <pthread.h>
#include <sys/qos.h>
#endif

namespace penta {

RTLogger::RTLogger()
    : writeIndex_(0), readIndex_(0), minLevel_(LogLevel::Info),
      running_(false) {}

RTLogger::~RTLogger() { stop(); }

void RTLogger::logRT(LogLevel level, const char *message) noexcept {
  if (level < minLevel_.load(std::memory_order_relaxed)) {
    return;
  }

  size_t writeIdx = writeIndex_.load(std::memory_order_relaxed);
  size_t nextIdx = (writeIdx + 1) % kQueueSize;

  // Check if queue is full
  if (nextIdx == readIndex_.load(std::memory_order_acquire)) {
    return; // Drop message
  }

  auto &msg = messageQueue_[writeIdx];
  msg.level = level;

  // Copy message (RT-safe string copy)
  size_t len = 0;
  while (message[len] && len < kMaxMessageSize - 1) {
    msg.text[len] = message[len];
    ++len;
  }
  msg.text[len] = '\0';

  msg.ready.store(true, std::memory_order_release);
  writeIndex_.store(nextIdx, std::memory_order_release);
}

void RTLogger::log(LogLevel level, const std::string &message) {
  logRT(level, message.c_str());
}

void RTLogger::start() {
  bool expected = false;
  if (!running_.compare_exchange_strong(expected, true,
                                        std::memory_order_acq_rel)) {
    return;
  }

  if (processingThread_.joinable()) {
    processingThread_.join();
  }

  processingThread_ = std::thread(&RTLogger::processingThread, this);
}

void RTLogger::stop() {
  const bool wasRunning = running_.exchange(false, std::memory_order_acq_rel);
  if (!wasRunning && !processingThread_.joinable()) {
    return;
  }
  if (processingThread_.joinable()) {
    processingThread_.join();
  }
}

void RTLogger::processingThread() {
#if defined(__APPLE__)
  // Pin to utility QoS so this thread does not steal realtime audio headroom.
  pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
#endif
  while (running_.load(std::memory_order_relaxed)) {
    size_t readIdx = readIndex_.load(std::memory_order_relaxed);

    if (readIdx == writeIndex_.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }

    auto &msg = messageQueue_[readIdx];
    if (!msg.ready.load(std::memory_order_acquire)) {
      continue;
    }

    // Process message
    const char *levelStr = "";
    switch (msg.level) {
    case LogLevel::Debug:
      levelStr = "DEBUG";
      break;
    case LogLevel::Info:
      levelStr = "INFO ";
      break;
    case LogLevel::Warning:
      levelStr = "WARN ";
      break;
    case LogLevel::Error:
      levelStr = "ERROR";
      break;
    }

    std::cout << "[" << levelStr << "] " << msg.text.data() << std::endl;

    msg.ready.store(false, std::memory_order_release);
    readIndex_.store((readIdx + 1) % kQueueSize, std::memory_order_release);
  }
}

RTLogger &getLogger() {
  static RTLogger instance;
  static std::once_flag startFlag;
  std::call_once(startFlag, [&] { instance.start(); });
  return instance;
}

} // namespace penta
