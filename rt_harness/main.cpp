/**
 * Headless RT harness: drives engine at fixed buffer size/sample rate,
 * measures per-callback duration (P50/P90/P99) and writes callback_stats.json.
 * See docs/research/MULTIMODAL_IMPLEMENTATIONS_PLAN.md section 3.3.
 */
#include <engine.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kDefaultSampleRate = 48000u;
constexpr uint32_t kDefaultBlockSize = 256u;
constexpr uint32_t kDefaultNumChannels = 2u;
constexpr double kDefaultDurationSec = 30.0;
constexpr char kDefaultOutput[] = "callback_stats.json";

struct Options {
  double duration_sec = kDefaultDurationSec;
  std::string output_path = kDefaultOutput;
  uint32_t sample_rate = kDefaultSampleRate;
  uint32_t block_size = kDefaultBlockSize;
  uint32_t num_channels = kDefaultNumChannels;
};

Options parse_args(int argc, char* argv[]) {
  Options o;
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--duration" && i + 1 < argc) {
      o.duration_sec = std::stod(argv[++i]);
    } else if (arg == "--output" && i + 1 < argc) {
      o.output_path = argv[++i];
    } else if (arg == "--sr" && i + 1 < argc) {
      o.sample_rate = static_cast<uint32_t>(std::stoul(argv[++i]));
    } else if (arg == "--block" && i + 1 < argc) {
      o.block_size = static_cast<uint32_t>(std::stoul(argv[++i]));
    } else if (arg == "--channels" && i + 1 < argc) {
      o.num_channels = static_cast<uint32_t>(std::stoul(argv[++i]));
    } else if (arg == "--help" || arg == "-h") {
      std::cerr
          << "Usage: rt_harness [--duration SEC] [--output PATH] [--sr RATE] "
             "[--block SIZE] [--channels N]\n"
          << "  --duration  Wall-clock run time in seconds (default: 30)\n"
          << "  --output    Output JSON path (default: callback_stats.json)\n"
          << "  --sr        Sample rate (default: 48000)\n"
          << "  --block     Block size in samples (default: 256)\n"
          << "  --channels  Number of channels (default: 2)\n";
      std::exit(0);
    }
  }
  return o;
}

double percentile_sorted(const std::vector<double>& sorted, double p) {
  if (sorted.empty()) return 0.0;
  const double idx = p * (static_cast<double>(sorted.size()) - 1.0);
  const size_t lo = static_cast<size_t>(idx);
  const size_t hi = lo + 1;
  if (hi >= sorted.size()) return sorted.back();
  const double frac = idx - static_cast<double>(lo);
  return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

void write_callback_stats(const std::string& path,
                          const std::vector<double>& us,
                          double duration_sec,
                          uint32_t buffer_size,
                          uint32_t sample_rate) {
  if (us.empty()) return;
  std::vector<double> sorted = us;
  std::sort(sorted.begin(), sorted.end());
  const double p50 = percentile_sorted(sorted, 0.50);
  const double p90 = percentile_sorted(sorted, 0.90);
  const double p99 = percentile_sorted(sorted, 0.99);
  const double min_us = sorted.front();
  const double max_us = sorted.back();
  const double sum = std::accumulate(us.begin(), us.end(), 0.0);
  const double mean_us = sum / static_cast<double>(us.size());

  std::ofstream f(path);
  if (!f) {
    std::cerr << "rt_harness: failed to open " << path << " for write\n";
    std::exit(1);
  }
  f << "{\n";
  f << "  \"p50_us\": " << p50 << ",\n";
  f << "  \"p90_us\": " << p90 << ",\n";
  f << "  \"p99_us\": " << p99 << ",\n";
  f << "  \"min_us\": " << min_us << ",\n";
  f << "  \"max_us\": " << max_us << ",\n";
  f << "  \"mean_us\": " << mean_us << ",\n";
  f << "  \"num_callbacks\": " << us.size() << ",\n";
  f << "  \"duration_sec\": " << duration_sec << ",\n";
  f << "  \"buffer_size\": " << buffer_size << ",\n";
  f << "  \"sample_rate\": " << sample_rate << "\n";
  f << "}\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  const Options opts = parse_args(argc, argv);
  if (opts.duration_sec <= 0.0 || opts.sample_rate == 0 ||
      opts.block_size == 0 || opts.num_channels == 0) {
    std::cerr << "rt_harness: invalid options\n";
    return 1;
  }

  kmidi_engine_config_t config = {};
  config.sample_rate = opts.sample_rate;
  config.max_block_size = opts.block_size;
  config.num_channels = opts.num_channels;

  kmidi_engine_t* engine = kmidi_engine_create(&config);
  if (!engine) {
    std::cerr << "rt_harness: kmidi_engine_create failed\n";
    return 1;
  }

  if (kmidi_engine_prepare(engine, opts.sample_rate, opts.block_size) != 0) {
    std::cerr << "rt_harness: kmidi_engine_prepare failed\n";
    kmidi_engine_destroy(engine);
    return 1;
  }

  // Allocate one block per channel (contiguous); then float* array for process.
  std::vector<std::vector<float>> channel_buffers(opts.num_channels);
  for (uint32_t c = 0; c < opts.num_channels; c++) {
    channel_buffers[c].resize(opts.block_size, 0.0f);
  }
  std::vector<float*> channel_ptrs(opts.num_channels);
  for (uint32_t c = 0; c < opts.num_channels; c++) {
    channel_ptrs[c] = channel_buffers[c].data();
  }

  const auto start_wall = std::chrono::steady_clock::now();
  std::vector<double> callback_us;
  callback_us.reserve(
      static_cast<size_t>(opts.duration_sec * opts.sample_rate / opts.block_size) +
      1000u);

  while (true) {
    auto t0 = std::chrono::high_resolution_clock::now();
    int r = kmidi_engine_process(engine, channel_ptrs.data(), opts.num_channels,
                                 opts.block_size);
    auto t1 = std::chrono::high_resolution_clock::now();
    if (r != 0) {
      std::cerr << "rt_harness: kmidi_engine_process failed\n";
      kmidi_engine_destroy(engine);
      return 1;
    }
    const auto us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    callback_us.push_back(us);

    const auto elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start_wall)
            .count();
    if (elapsed >= opts.duration_sec) break;
  }

  const auto end_wall = std::chrono::steady_clock::now();
  const double duration_sec =
      std::chrono::duration<double>(end_wall - start_wall).count();

  kmidi_engine_destroy(engine);

  write_callback_stats(opts.output_path, callback_us, duration_sec,
                      opts.block_size, opts.sample_rate);
  std::cout << "rt_harness: wrote " << callback_us.size() << " callbacks to "
            << opts.output_path << "\n";
  return 0;
}
