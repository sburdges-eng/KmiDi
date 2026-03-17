# Core ML LLM Runner (stateful KV-cache)

Swift executable that loads a compiled **mlmodelc** (exported with ExecuTorch's `export_llama_lib.py` using `--coreml-enable-state`, `--use_kv_cache`, etc.), threads **state output at step t** back in as **state input at step t+1**, and decodes **greedily** for minimal latency.

Target: **sub-15 ms/token** on M4 hardware when the model is compiled and state is managed by the Neural Engine.

## Prerequisites

- **macOS 15+** (stateful Core ML prediction API)
- Xcode Command Line Tools or Xcode

## Build

```bash
cd tools/coreml_llm_runner
swift build -c release
```

Binary: `.build/release/CoreMLLMRunner`

## Export flow (before using the runner)

1. Export the LLM to Core ML using the KmiDi wrapper script (requires ExecuTorch elsewhere):

   ```bash
   EXECUTORCH_DIR=/path/to/executorch python scripts/export_llm_coreml.py \
     --model-path /path/to/llama-checkpoint \
     --output-dir ./coreml_llm_export
   ```

2. Compile the generated `.mlpackage` to **mlmodelc**:

   ```bash
   xcrun coremlcompiler compile Model.mlpackage ./build/
   ```

3. Run the Swift runner on the compiled directory:

   ```bash
   .build/release/CoreMLLMRunner ./build/Model.mlmodelc --max-tokens 64 --timing
   ```

## Usage

```text
CoreMLLMRunner <path-to-mlmodelc> [--max-tokens N] [--timing]
```

- **path-to-mlmodelc**: Path to the compiled model directory (e.g. `Model.mlmodelc`).
- **--max-tokens N**: Maximum decode steps (default 32).
- **--timing**: Print ms per token and average ms/token to stderr.

## State threading

The runner discovers **state** and **token/logits** inputs and outputs from the model's description. It assumes:

- State output names and state input names correspond by **index** (first state out → first state in).
- Token input and logits output are the remaining primary input/output.

At each step, state outputs from the previous prediction are fed back as state inputs so the KV cache stays in hardware-bound memory (no copy back to host). This achieves the sub-15 ms/token target on M4 when the Core ML backend uses the Neural Engine.

If your export uses different naming, inspect the model in Xcode (model viewer) and adjust the discovery logic in `main.swift`.

## Prefill vs decode-only

The runner is **decode-only**: it runs one token per step in a loop. It does not perform a single batched prefill (full prompt in one forward pass). If your compiled model exposes a prefill-style API (batch `input_ids` → logits + state in one call), you can add an optional prefill path in the runner to reduce first-token latency; until then, the first token is produced by the same single-token step as the rest.

## Integration with Tauri

For on-device LLM in the KmiDi desktop app, Tauri can spawn this runner as a subprocess or call into a Swift library; the runner is kept as a standalone tool for verification and latency measurement.
