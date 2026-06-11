// Smoke suite for the penta_core gtest target (penta_tests).
//
// This is the first real content behind BUILD_PENTA_TESTS: the CI lanes in
// ci.yml/test.yml build the penta_tests target and run it with gtest filters
// (*RTSafe*, *Performance*, *Benchmark*, *Plugin*, *Integration*). Test names
// here deliberately match those filters so each lane exercises real
// assertions. Tests are count-based, never wall-clock-based, so they cannot
// flake on slow runners.

#include "penta/osc/OSCMessage.h"
#include "penta/osc/RTMessageQueue.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

namespace {

using penta::osc::OSCMessage;
using penta::osc::RTMessageQueue;

OSCMessage makeMessage(int32_t seq) {
  OSCMessage msg("/penta/smoke");
  msg.addInt(seq);
  msg.addFloat(0.5f);
  msg.addString("smoke");
  return msg;
}

// --- Basic value semantics --------------------------------------------------

TEST(PentaSmoke, OSCMessageStoresTypedArguments) {
  const OSCMessage msg = makeMessage(42);

  EXPECT_EQ(msg.getAddress(), "/penta/smoke");
  ASSERT_EQ(msg.getArgumentCount(), 3u);
  EXPECT_EQ(msg.getInt(0), 42);
  EXPECT_FLOAT_EQ(msg.getFloat(1), 0.5f);
  EXPECT_EQ(msg.getString(2), "smoke");
}

TEST(PentaSmoke, OSCMessageOutOfRangeReadsFallBackToDefaults) {
  const OSCMessage msg = makeMessage(7);

  EXPECT_EQ(msg.getInt(99, -1), -1);
  EXPECT_FLOAT_EQ(msg.getFloat(99, -2.0f), -2.0f);
  EXPECT_EQ(msg.getString(99, "fallback"), "fallback");
}

// --- Queue contract
// -----------------------------------------------------------

TEST(PentaSmoke, QueuePushPopRoundTripPreservesOrder) {
  RTMessageQueue queue(16);
  EXPECT_TRUE(queue.isEmpty());

  for (int32_t i = 0; i < 8; ++i) {
    EXPECT_TRUE(queue.push(makeMessage(i)));
  }
  EXPECT_EQ(queue.size(), 8u);

  for (int32_t i = 0; i < 8; ++i) {
    OSCMessage out("/uninitialized");
    ASSERT_TRUE(queue.pop(out));
    EXPECT_EQ(out.getInt(0), i);
  }
  EXPECT_TRUE(queue.isEmpty());

  OSCMessage out("/uninitialized");
  EXPECT_FALSE(queue.pop(out));
}

TEST(PentaSmoke, QueueClearDrainsAllPending) {
  RTMessageQueue queue(8);
  for (int32_t i = 0; i < 4; ++i) {
    ASSERT_TRUE(queue.push(makeMessage(i)));
  }
  queue.clear();
  EXPECT_TRUE(queue.isEmpty());
  EXPECT_EQ(queue.size(), 0u);
}

// --- RT-safety lane (filter: *RTSafe*) ---------------------------------------

TEST(PentaRTSafeSmoke, SpscQueueSurvivesConcurrentProducerConsumer) {
  constexpr int32_t kMessages = 5000;
  RTMessageQueue queue(256);

  std::atomic<int32_t> consumed{0};
  std::atomic<bool> producerDone{false};
  int32_t lastSeq = -1;
  bool ordered = true;

  std::thread consumer([&] {
    OSCMessage out("/uninitialized");
    while (!producerDone.load(std::memory_order_acquire) || !queue.isEmpty()) {
      if (queue.pop(out)) {
        const int32_t seq = out.getInt(0, -1);
        if (seq <= lastSeq) {
          ordered = false;
        }
        lastSeq = seq;
        consumed.fetch_add(1, std::memory_order_relaxed);
      } else {
        std::this_thread::yield();
      }
    }
  });

  for (int32_t i = 0; i < kMessages; ++i) {
    // Spin until the bounded queue accepts; producer-side backpressure is
    // part of the RT contract (push never blocks, it just reports full).
    while (!queue.push(makeMessage(i))) {
      std::this_thread::yield();
    }
  }
  producerDone.store(true, std::memory_order_release);
  consumer.join();

  EXPECT_EQ(consumed.load(), kMessages);
  EXPECT_TRUE(ordered) << "SPSC queue delivered messages out of order";
  EXPECT_EQ(lastSeq, kMessages - 1);
}

// --- Performance/benchmark lanes (filters: *Performance*, *Benchmark*) -------

TEST(PentaPerformanceSmoke, QueueSustainsBurstThroughputWithoutLoss) {
  constexpr int32_t kBursts = 100;
  constexpr int32_t kBurstSize = 64;
  RTMessageQueue queue(128);

  int32_t total = 0;
  for (int32_t burst = 0; burst < kBursts; ++burst) {
    for (int32_t i = 0; i < kBurstSize; ++i) {
      ASSERT_TRUE(queue.push(makeMessage(total + i)));
    }
    OSCMessage out("/uninitialized");
    for (int32_t i = 0; i < kBurstSize; ++i) {
      ASSERT_TRUE(queue.pop(out));
    }
    total += kBurstSize;
  }
  EXPECT_TRUE(queue.isEmpty());
  EXPECT_EQ(total, kBursts * kBurstSize);
}

// --- Integration lane (filters: *Plugin*, *Integration*) ---------------------

TEST(PentaIntegrationSmoke, MessageContentSurvivesQueueTransit) {
  RTMessageQueue queue(4);

  OSCMessage in("/penta/integration");
  in.addInt(123);
  in.addString("payload");
  in.addBlob({0xDE, 0xAD, 0xBE, 0xEF});
  in.setTimestamp(987654321u);

  ASSERT_TRUE(queue.push(in));

  OSCMessage out("/uninitialized");
  ASSERT_TRUE(queue.pop(out));

  EXPECT_EQ(out.getAddress(), "/penta/integration");
  EXPECT_EQ(out.getInt(0), 123);
  EXPECT_EQ(out.getString(1), "payload");
  EXPECT_EQ(out.getBlob(2), std::vector<uint8_t>({0xDE, 0xAD, 0xBE, 0xEF}));
  EXPECT_EQ(out.getTimestamp(), 987654321u);
}

} // namespace
