# RT Harness

Headless real-time callback harness for the KmiDi engine (C ABI). Runs the engine at a fixed buffer size and sample rate, measures per-callback duration (P50/P90/P99), and writes `callback_stats.json`.

See [docs/research/MULTIMODAL_IMPLEMENTATIONS_PLAN.md](../docs/research/MULTIMODAL_IMPLEMENTATIONS_PLAN.md) section 3.3.

## Build

From repo root:

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_RT_HARNESS=ON
cmake --build build --target rt_harness
```

## Run

```bash
./build/rt_harness/rt_harness --duration 30 --output callback_stats.json
```

Options:

- `--duration SEC`  Wall-clock run time (default: 30).
- `--output PATH`   Output JSON path (default: callback_stats.json).
- `--sr RATE`       Sample rate (default: 48000).
- `--block SIZE`    Block size in samples (default: 256).
- `--channels N`    Number of channels (default: 2).

## CI

The workflow builds `engine` and `rt_harness`, runs the harness for 30 s, then fails if `p90_us` in the output JSON exceeds the configured threshold (e.g. `RT_HARNESS_P90_US_MAX=500`).

## Golden preset

Default config is 48 kHz, block 256, 2 channels. A copy is documented in `fixtures/golden_preset.json` for reproducibility.
