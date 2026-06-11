// Lifecycle + ownership test for Kelly::ML::MultiModelProcessor and
// AsyncMLPipeline (src/ml/MultiModelProcessor.{h,cpp}).
//
// Standalone main (no Catch2), mirroring test_rt_event_scheduler.cpp style.
// Links KellyCore, which provides juce_core transitively.
//
// Coverage:
//   1. Fallback initialization without a models directory
//   2. Synchronous infer() output sizing per model spec
//   3. Input-size-mismatch safety on infer()
//   4. submitFeatures() busy rejection when no worker drains the slot
//   5. Async submit -> result roundtrip in fallback mode
//   6. Stale request-id getResult() returns invalid and does not consume
//   7. Restart safety: start -> stop -> start on one pipeline instance
//   8. Idempotent double start / double stop / stop-before-start
//   9. Model enable toggling observed by the full pipeline
//  10. Concurrent start/stop/submit torture across threads, then a clean
//      roundtrip proving the pipeline survived
//  11. Destruction while the worker is running (dtor performs the stop)

#include "ml/MultiModelProcessor.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char *label) {
  if (!condition) {
    std::printf("FAIL: %s\n", label);
    ++g_failures;
  }
}

using Kelly::ML::AsyncMLPipeline;
using Kelly::ML::InferenceResult;
using Kelly::ML::MODEL_SPECS;
using Kelly::ML::ModelType;
using Kelly::ML::MultiModelProcessor;

// Absolute path that must not exist so every model falls back to heuristics.
const juce::File kMissingModelsDir("/tmp/kmidi_test_no_models_dir_d41d8cd9");

std::array<float, 128> makeFeatures(float value) {
  std::array<float, 128> features{};
  features.fill(value);
  return features;
}

bool waitForResult(const AsyncMLPipeline &pipeline, uint64_t requestId,
                   int maxMillis) {
  for (int i = 0; i < maxMillis; ++i) {
    if (pipeline.hasResult(requestId)) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return pipeline.hasResult(requestId);
}

bool anyNonZero(const float *data, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    if (std::fabs(data[i]) > 1e-6f) {
      return true;
    }
  }
  return false;
}

// 1. Fallback initialization ------------------------------------------------

void test_fallback_initialize() {
  MultiModelProcessor processor;
  check(!processor.isInitialized(), "fresh processor not initialized");
  check(processor.initialize(kMissingModelsDir),
        "initialize succeeds in fallback mode");
  check(processor.isInitialized(), "processor reports initialized");
  check(processor.getTotalParams() > 0, "total params positive");
  check(processor.getTotalMemoryKB() > 0, "total memory positive");
}

// 2. infer() output sizing ---------------------------------------------------

void test_infer_output_sizes() {
  MultiModelProcessor processor;
  processor.initialize(kMissingModelsDir);

  for (size_t i = 0; i < static_cast<size_t>(ModelType::COUNT); ++i) {
    const auto &spec = MODEL_SPECS[i];
    std::vector<float> input(spec.inputSize, 0.5f);
    auto output = processor.infer(static_cast<ModelType>(i), input);
    check(output.size() == spec.outputSize, "infer output matches spec size");
  }
}

// 3. infer() size-mismatch safety ---------------------------------------------

void test_infer_size_mismatch_safe() {
  MultiModelProcessor processor;
  processor.initialize(kMissingModelsDir);

  std::vector<float> wrongInput(3, 1.0f);
  auto output = processor.infer(ModelType::EmotionRecognizer, wrongInput);
  check(output.size() ==
            MODEL_SPECS[static_cast<size_t>(ModelType::EmotionRecognizer)]
                .outputSize,
        "mismatched input still yields spec-sized output");
}

// 4. Busy rejection without a worker ------------------------------------------

void test_submit_busy_without_worker() {
  MultiModelProcessor processor;
  processor.initialize(kMissingModelsDir);

  AsyncMLPipeline pipeline(processor);
  const auto features = makeFeatures(0.25f);

  const uint64_t first = pipeline.submitFeatures(features);
  check(first != 0, "first submission accepted");

  const uint64_t second = pipeline.submitFeatures(features);
  check(second == 0, "second submission rejected while slot is pending");

  check(!pipeline.hasResult(), "no result without a worker");
  // Pipeline destroyed without ever starting: must not crash or hang.
}

// 5. Async roundtrip
// -----------------------------------------------------------

void test_async_roundtrip() {
  MultiModelProcessor processor;
  processor.initialize(kMissingModelsDir);

  AsyncMLPipeline pipeline(processor);
  pipeline.start();

  const uint64_t id = pipeline.submitFeatures(makeFeatures(0.5f));
  check(id != 0, "roundtrip submission accepted");
  check(waitForResult(pipeline, id, 5000), "roundtrip result arrives");

  const InferenceResult result = pipeline.getResult(id);
  check(result.valid, "roundtrip result valid");
  check(anyNonZero(result.emotionEmbedding.data(),
                   result.emotionEmbedding.size()),
        "fallback emotion embedding non-zero");
  check(!pipeline.hasResult(), "result consumed after getResult");

  pipeline.stop();
}

// 6. Stale request id
// ----------------------------------------------------------

void test_stale_id_invalid() {
  MultiModelProcessor processor;
  processor.initialize(kMissingModelsDir);

  AsyncMLPipeline pipeline(processor);
  pipeline.start();

  const uint64_t id = pipeline.submitFeatures(makeFeatures(0.5f));
  check(waitForResult(pipeline, id, 5000), "stale-id test result arrives");

  const InferenceResult stale = pipeline.getResult(id + 12345);
  check(!stale.valid, "stale id yields invalid result");
  check(pipeline.hasResult(id), "stale-id getResult does not consume");

  const InferenceResult real = pipeline.getResult(id);
  check(real.valid, "real id still retrievable after stale query");

  pipeline.stop();
}

// 7. Restart safety
// -------------------------------------------------------------

void test_restart_lifecycle() {
  MultiModelProcessor processor;
  processor.initialize(kMissingModelsDir);

  AsyncMLPipeline pipeline(processor);

  for (int cycle = 0; cycle < 2; ++cycle) {
    pipeline.start();
    const uint64_t id = pipeline.submitFeatures(makeFeatures(0.4f));
    check(id != 0, "restart cycle submission accepted");
    check(waitForResult(pipeline, id, 5000), "restart cycle result arrives");
    check(pipeline.getResult(id).valid, "restart cycle result valid");
    pipeline.stop();
  }
}

// 8. Idempotent lifecycle calls
// ---------------------------------------------------

void test_idempotent_start_stop() {
  MultiModelProcessor processor;
  processor.initialize(kMissingModelsDir);

  AsyncMLPipeline pipeline(processor);

  pipeline.stop(); // stop before any start: must be a safe no-op
  pipeline.start();
  pipeline.start(); // double start: must not spawn a second worker

  const uint64_t id = pipeline.submitFeatures(makeFeatures(0.3f));
  check(id != 0, "submission accepted after double start");
  check(waitForResult(pipeline, id, 5000), "result arrives after double start");
  check(pipeline.getResult(id).valid, "result valid after double start");

  pipeline.stop();
  pipeline.stop(); // double stop: must be a safe no-op
}

// 9. Enable toggling through the pipeline
// -----------------------------------------

void test_enabled_toggle_pipeline() {
  MultiModelProcessor processor;
  processor.initialize(kMissingModelsDir);

  check(processor.isModelEnabled(ModelType::EmotionRecognizer),
        "models enabled by default");

  processor.setModelEnabled(ModelType::EmotionRecognizer, false);
  check(!processor.isModelEnabled(ModelType::EmotionRecognizer),
        "disable visible through isModelEnabled");

  const InferenceResult disabled =
      processor.runFullPipeline(makeFeatures(0.5f));
  check(disabled.valid, "pipeline result valid with disabled head");
  check(!anyNonZero(disabled.emotionEmbedding.data(),
                    disabled.emotionEmbedding.size()),
        "disabled emotion head leaves embedding zeroed");

  processor.setModelEnabled(ModelType::EmotionRecognizer, true);
  const InferenceResult enabled = processor.runFullPipeline(makeFeatures(0.5f));
  check(anyNonZero(enabled.emotionEmbedding.data(),
                   enabled.emotionEmbedding.size()),
        "re-enabled emotion head produces output");
}

// 10. Concurrent lifecycle torture
// --------------------------------------------------

void test_concurrent_lifecycle_torture() {
  MultiModelProcessor processor;
  processor.initialize(kMissingModelsDir);

  AsyncMLPipeline pipeline(processor);

  constexpr int kThreads = 4;
  constexpr int kIterations = 50;
  std::atomic<bool> go{false};

  std::vector<std::thread> workers;
  workers.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    workers.emplace_back([&pipeline, &go, t]() {
      while (!go.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      const auto features = makeFeatures(0.1f * static_cast<float>(t + 1));
      for (int i = 0; i < kIterations; ++i) {
        if (t % 2 == 0) {
          pipeline.start();
          pipeline.submitFeatures(features);
        } else {
          pipeline.stop();
        }
        std::this_thread::yield();
      }
    });
  }

  go.store(true, std::memory_order_release);
  for (auto &worker : workers) {
    worker.join();
  }

  pipeline.stop();

  // The pipeline must still be fully functional after the storm.
  pipeline.start();
  uint64_t id = 0;
  for (int attempt = 0; attempt < 1000 && id == 0; ++attempt) {
    id = pipeline.submitFeatures(makeFeatures(0.5f));
    if (id == 0) {
      if (pipeline.hasResult()) {
        (void)pipeline.getResult(); // drain a leftover torture result
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  check(id != 0, "post-torture submission accepted");
  check(waitForResult(pipeline, id, 5000), "post-torture result arrives");
  check(pipeline.getResult(id).valid, "post-torture result valid");
  pipeline.stop();
}

// 11. Destructor stops a running worker
// ----------------------------------------------

void test_destructor_stops_running_worker() {
  MultiModelProcessor processor;
  processor.initialize(kMissingModelsDir);

  {
    AsyncMLPipeline pipeline(processor);
    pipeline.start();
    pipeline.submitFeatures(makeFeatures(0.2f));
    // Scope exit destroys the pipeline while the worker may be mid-inference.
  }

  check(true, "destructor completed with a running worker");
}

} // namespace

int main() {
  // Unbuffered output so a crash mid-suite still shows which test was running.
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("test_multi_model_processor: start\n");

  struct Step {
    const char *name;
    void (*fn)();
  };
  const Step steps[] = {
      {"fallback_initialize", test_fallback_initialize},
      {"infer_output_sizes", test_infer_output_sizes},
      {"infer_size_mismatch_safe", test_infer_size_mismatch_safe},
      {"submit_busy_without_worker", test_submit_busy_without_worker},
      {"async_roundtrip", test_async_roundtrip},
      {"stale_id_invalid", test_stale_id_invalid},
      {"restart_lifecycle", test_restart_lifecycle},
      {"idempotent_start_stop", test_idempotent_start_stop},
      {"enabled_toggle_pipeline", test_enabled_toggle_pipeline},
      {"concurrent_lifecycle_torture", test_concurrent_lifecycle_torture},
      {"destructor_stops_running_worker", test_destructor_stops_running_worker},
  };
  for (const auto &step : steps) {
    std::printf("-- %s\n", step.name);
    step.fn();
  }

  if (g_failures == 0) {
    std::printf("test_multi_model_processor: ALL PASS\n");
    return 0;
  }
  std::printf("test_multi_model_processor: %d FAILURE(S)\n", g_failures);
  return 1;
}
